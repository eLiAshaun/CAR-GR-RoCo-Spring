import math

import torch
import torch.nn.functional as F

from model.waft_a2 import WAFTv2
from model.modules import (
    CorrespondenceResidualInjection,
    MultiScaleResidualInjection,
    UncertaintyDampedGate,
    ZIRA,
)
from utils.utils import Padder, bilinear_sampler, coords_grid


class WAFTRX(WAFTv2):
    """Independent WAFT extension; the parent WAFT-A2 source stays untouched."""

    def __init__(self, args):
        super().__init__(args)
        if getattr(args, "rx_zira", False):
            self.zira = ZIRA(self.iter_dim)
        if getattr(args, "rx_msri", False):
            self.msri = MultiScaleResidualInjection(
                target_channels=self.iter_dim,
                use_s4=getattr(args, "rx_msri_s4", False),
            )
        if getattr(args, "rx_cri", False):
            self.cri = CorrespondenceResidualInjection(self.iter_dim)
        if getattr(args, "rx_udg", False):
            self.udg = UncertaintyDampedGate(self.iter_dim, info_dim=4)

    def build_frame_feature(self, image, pretrained_feature):
        image_pyramid = self.fnet(image)
        base = self.fmap_conv(torch.cat([pretrained_feature, image_pyramid[0]], dim=1))
        if hasattr(self, "msri"):
            base = self.msri(base, image_pyramid)
        if hasattr(self, "zira"):
            base = self.zira(base)
        return base

    def forward(self, image1, image2, iters=None, flow_gt=None):
        if iters is None:
            iters = self.args.iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        flow_predictions = []
        info_predictions = []
        N, _, H, W = image1.shape
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_2x = self.build_frame_feature(image1, fmap1_pretrain)
        fmap2_2x = self.build_frame_feature(image2, fmap2_pretrain)
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2).to(image1.device)
        for itr in range(iters):
            flow_2x = flow_2x.detach()
            coords2 = (coords_grid(N, H // 2, W // 2, device=image1.device) + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            if hasattr(self, "cri"):
                refine_inp = self.cri(refine_inp, fmap1_2x, warp_2x)
            refine_outs = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_outs["out"], net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            if hasattr(self, "udg"):
                flow_2x = self.udg(flow_2x, flow_update[:, :2], net, flow_update[:, 2:])
            else:
                flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)

        for i in range(len(info_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            info_predictions[i] = padder.unpad(info_predictions[i])

        output = {"flow": flow_predictions, "info": info_predictions}
        if getattr(self.args, "return_features", False):
            output["feature"] = [fmap1_2x, fmap2_2x]
        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (
                    torch.exp(-log_b).unsqueeze(1)
                )
                term1 = weight - math.log(2) - log_b
                nf_predictions.append(
                    torch.logsumexp(weight, dim=1, keepdim=True)
                    - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                )
            output["nf"] = nf_predictions
        return output
