"""Native adapters for the isolated WAFT/MEMFOF Spring evaluation."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / "models.json").read_text())


def get_amp_dtype(precision: str):
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unknown precision: {precision}")


def validate_pair(image1: np.ndarray, image2: np.ndarray) -> tuple[int, int]:
    if image1.ndim != 3 or image1.shape[2] != 3 or image2.ndim != 3 or image2.shape[2] != 3:
        raise ValueError("expected two HxWx3 color images")
    if image1.shape != image2.shape:
        raise ValueError("pair images must have the same shape")
    if image1.dtype != np.uint8 or image2.dtype != np.uint8:
        raise ValueError("adapter inputs must be uint8")
    return image1.shape[:2]


def validate_prediction(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    prediction = np.asarray(prediction)
    if prediction.shape != (*shape, 2):
        raise ValueError(f"expected prediction shape {(*shape, 2)}, got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("model returned non-finite flow")
    return prediction.astype(np.float32, copy=False)


def select_memfof_forward(output) -> np.ndarray:
    """Select middle-to-next flow from [B, temporal_direction, xy, H, W]."""
    flow = output[0, 1]
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().float().cpu().permute(1, 2, 0).numpy()
    else:
        flow = np.asarray(flow).transpose(1, 2, 0)
    return flow.astype(np.float32, copy=False)


def _rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1])).permute(2, 0, 1).float()[None].to(device)


def _extract_checkpoint(model_id: str, spec: dict) -> Path:
    if "checkpoint" in spec:
        path = ROOT / spec["checkpoint"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    archive = ROOT / spec["archive"]
    if not archive.is_file():
        raise FileNotFoundError(archive)
    target = ROOT / "weights" / "waft" / "extracted" / f"{model_id}.pth"
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        try:
            source = handle.open(spec["archive_member"])
        except KeyError as exc:
            raise FileNotFoundError(spec["archive_member"]) from exc
        with source, target.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
    return target


def _waft_model(model_id: str, device: torch.device):
    spec = MODELS[model_id]
    source = ROOT / "sources" / "WAFT"
    config = json.loads((ROOT / spec["config"]).read_text())
    checkpoint = _extract_checkpoint(model_id, spec)
    os.chdir(source)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import timm
    from model.waft_a1 import ViTWarpV8
    from model.waft_a2 import WAFTv2

    real_create_model = timm.create_model

    def local_create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return real_create_model(*args, **kwargs)

    old_create_model = timm.create_model
    timm.create_model = local_create_model
    try:
        args = SimpleNamespace(**config)
        if model_id == "waft_dav2_a1":
            import model.waft_a1 as waft_a1

            original = waft_a1.DepthAnythingFeature
            waft_a1.DepthAnythingFeature = lambda encoder="vits", pretrained=True: original(
                encoder, pretrained=False
            )
            try:
                model = ViTWarpV8(args)
            finally:
                waft_a1.DepthAnythingFeature = original
        elif model_id == "waft_dav2_a2":
            import model.waft_a2 as waft_a2

            original = waft_a2.DepthAnythingFeature
            waft_a2.DepthAnythingFeature = lambda model_name="vits", pretrained=True, lvl=-3: original(
                model_name, pretrained=False, lvl=lvl
            )
            try:
                model = WAFTv2(args)
            finally:
                waft_a2.DepthAnythingFeature = original
        elif model_id == "waft_twins_a2":
            model = WAFTv2(args)
        elif model_id == "waft_dinov3_a2":
            import model.backbone.dinov3 as waft_dinov3

            original_load = torch.hub.load

            def local_hub_load(repo_or_dir, model_name, *load_args, **kwargs):
                if kwargs.get("source") == "local" and "dinov3" in str(repo_or_dir):
                    kwargs["pretrained"] = False
                    kwargs.pop("weights", None)
                return original_load(repo_or_dir, model_name, *load_args, **kwargs)

            torch.hub.load = local_hub_load
            try:
                model = WAFTv2(args)
            finally:
                torch.hub.load = original_load
        else:
            raise ValueError(f"unknown WAFT model: {model_id}")
    finally:
        timm.create_model = old_create_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config, checkpoint


class WAFTPredictor:
    def __init__(self, model_id: str, device: torch.device, precision: str):
        self.model, self.config, self.checkpoint = _waft_model(model_id, device)
        self.device = device
        self.amp_dtype = get_amp_dtype(precision) if device.type == "cuda" else None
        self.precision = precision if self.amp_dtype is not None else "fp32"
        self.scale = float(self.config.get("scale", 0))

    def predict(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        height, width = validate_pair(image1, image2)
        first, second = _rgb_tensor(image1, self.device), _rgb_tensor(image2, self.device)
        if self.scale:
            factor = 2**self.scale
            first = F.interpolate(first, scale_factor=factor, mode="bilinear", align_corners=True)
            second = F.interpolate(second, scale_factor=factor, mode="bilinear", align_corners=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            flow = self.model(first, second)["flow"][-1][0]
        if self.scale:
            flow = F.interpolate(flow[None], size=(height, width), mode="bilinear", align_corners=True)[0]
            flow = flow * (0.5**self.scale)
        prediction = flow.detach().float().cpu().permute(1, 2, 0).numpy()
        return validate_prediction(prediction, (height, width))


def _memfof_model(device: torch.device):
    source = ROOT / "sources" / "memfof"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from memfof import MEMFOF
    from safetensors.torch import load_file

    spec = MODELS["memfof_spring"]
    config = json.loads((ROOT / spec["config"]).read_text())
    checkpoint = ROOT / spec["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MEMFOF(backbone_weights=None, **config)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model.to(device).eval(), config, checkpoint


class MEMFOFPredictor:
    def __init__(self, device: torch.device, precision: str):
        self.model, self.config, self.checkpoint = _memfof_model(device)
        self.device = device
        self.amp_dtype = get_amp_dtype(precision) if device.type == "cuda" else None
        self.precision = precision if self.amp_dtype is not None else "fp32"

    def predict_triplet(self, context: np.ndarray, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        height, width = validate_pair(image1, image2)
        if context.shape != image1.shape or context.dtype != np.uint8:
            raise ValueError("MEMFOF context must match the pair and be uint8")
        frames = torch.stack(
            [_rgb_tensor(frame, self.device)[0] for frame in (context, image1, image2)], dim=0
        )[None]
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            output = self.model(frames)
        return validate_prediction(select_memfof_forward(output["flow"][-1]), (height, width))

    def predict(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        return self.predict_triplet(image1, image1, image2)


def load_predictor(model_id: str, device: str | torch.device = "cuda", precision: str = "bf16"):
    if model_id not in MODELS:
        raise ValueError(f"unknown model id: {model_id}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    get_amp_dtype(precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if MODELS[model_id]["family"] == "waft":
        return WAFTPredictor(model_id, device, precision)
    return MEMFOFPredictor(device, precision)
