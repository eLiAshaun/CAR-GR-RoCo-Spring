"""Clean, paired-corruption, and group-robust losses for Stage4."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _valid(flow_gt, valid, max_flow=40000.0):
    magnitude = torch.linalg.vector_norm(flow_gt, dim=1)
    return (valid >= 0.5) & (magnitude < max_flow)


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1.0)


def sequence_loss_per_sample(
    output, flow_gt, valid, gamma=0.85, max_flow=40000.0,
):
    """The original WAFT NF sequence loss, reduced per sample."""
    if "nf" not in output:
        raise KeyError("output must contain nf when computing supervised loss")
    valid = _valid(flow_gt, valid, max_flow)
    result = flow_gt.new_zeros(flow_gt.shape[0])
    for index, nf in enumerate(output["nf"]):
        finite = torch.isfinite(nf[:, 0])
        mask = valid & finite
        result = result + gamma ** (len(output["nf"]) - index - 1) * _masked_mean(nf[:, 0], mask)
    return result


def final_epe_per_sample(output, flow_gt, valid, max_flow=40000.0):
    error = torch.linalg.vector_norm(output["flow"][-1] - flow_gt, dim=1)
    return _masked_mean(error, _valid(flow_gt, valid, max_flow))


def sequence_epe_per_sample(output, flow_gt, valid, gamma=0.85, epsilon=1e-3, max_flow=40000.0):
    """Metric-aligned EPE over every iterative prediction."""
    valid = _valid(flow_gt, valid, max_flow)
    result = flow_gt.new_zeros(flow_gt.shape[0])
    count = len(output["flow"])
    for index, prediction in enumerate(output["flow"]):
        error = torch.sqrt((prediction - flow_gt).square().sum(dim=1) + epsilon**2)
        result = result + gamma ** (count - index - 1) * _masked_mean(error, valid)
    return result


def charbonnier_per_sample(error, valid=None, epsilon=1e-3):
    value = torch.sqrt(error.square().sum(dim=1) + epsilon**2)
    if valid is None:
        return value.flatten(1).mean(1)
    return _masked_mean(value, valid)


def clean_supervised_loss(
    output, flow_gt, valid, gamma=0.85, eta_final_epe=0.05, epe_alpha=0.0,
):
    nf = sequence_loss_per_sample(output, flow_gt, valid, gamma)
    epe = final_epe_per_sample(output, flow_gt, valid)
    sequence_epe = sequence_epe_per_sample(output, flow_gt, valid, gamma)
    return {
        "per_sample": nf + eta_final_epe * epe + epe_alpha * sequence_epe,
        "nf": nf.mean(),
        "epe": epe.mean(),
        "sequence_epe": sequence_epe.mean(),
    }


def consistency_per_sample(corrupt_output, clean_target, valid, epsilon=1e-3):
    error = corrupt_output["flow"][-1] - clean_target.detach()
    return charbonnier_per_sample(error, _valid(clean_target, valid), epsilon)


def group_dro(per_sample, group_ids, temperature=0.3):
    """Smooth maximum over groups present in this batch."""
    group_ids = group_ids.to(device=per_sample.device, dtype=torch.long)
    present = torch.unique(group_ids[group_ids >= 0])
    if present.numel() == 0:
        return per_sample.mean(), {}
    means = {}
    values = []
    for group in present.tolist():
        value = per_sample[group_ids == group].mean()
        means[int(group)] = value
        values.append(value)
    stacked = torch.stack(values)
    return temperature * torch.logsumexp(stacked / temperature, dim=0), means


def soft_worst_corruption(
    per_sample,
    corruption_ids,
    beta=0.2,
    temperature=0.3,
    ema=None,
    ema_decay=0.9,
):
    """Macro mean plus a small smooth worst-corruption auxiliary."""
    corruption_ids = corruption_ids.to(device=per_sample.device, dtype=torch.long)
    present = torch.unique(corruption_ids[corruption_ids >= 0])
    if present.numel() == 0:
        return per_sample.mean(), {}, temperature
    means = {int(index): per_sample[corruption_ids == index].mean() for index in present.tolist()}
    current = torch.stack(list(means.values()))
    if ema is not None:
        with torch.no_grad():
            for index, value in means.items():
                ema[index] = ema_decay * ema[index] + (1.0 - ema_decay) * value.detach()
            scale = ema[present].mean().clamp_min(1e-3) * temperature
        temperature = float(scale.detach())
    temperature_tensor = current.new_tensor(temperature)
    macro = current.mean()
    smooth_worst = temperature_tensor * torch.logsumexp(
        current / temperature_tensor - math.log(float(current.numel())), dim=0
    )
    return (1.0 - beta) * macro + beta * smooth_worst, means, temperature


def l2_sp_loss(model, anchors):
    if not anchors:
        return next(model.parameters()).new_zeros(())
    terms = [
        (parameter - anchors[name]).square().mean()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in anchors
    ]
    return torch.stack(terms).mean() if terms else next(model.parameters()).new_zeros(())


def feature_consistency_per_sample(clean_feature, corrupt_feature, valid=None):
    clean_feature = F.normalize(clean_feature.detach(), dim=1, eps=1e-6)
    corrupt_feature = F.normalize(corrupt_feature, dim=1, eps=1e-6)
    value = 1.0 - (clean_feature * corrupt_feature).sum(dim=1)
    if valid is not None:
        valid = F.interpolate(valid[:, None].float(), size=value.shape[-2:], mode="nearest")[:, 0] > 0.5
    return _masked_mean(value, valid) if valid is not None else value.flatten(1).mean(1)


def gradient_cosine(clean_grads, robust_grads):
    pairs = [
        (clean, robust)
        for clean, robust in zip(clean_grads, robust_grads)
        if clean is not None and robust is not None
    ]
    if not pairs:
        return 0.0
    clean = torch.cat([item[0].detach().flatten() for item in pairs])
    robust = torch.cat([item[1].detach().flatten() for item in pairs])
    return float(F.cosine_similarity(clean[None], robust[None], dim=1).item())


def project_robust_gradients(clean_grads, robust_grads, project_mask):
    """Project only selected robust gradients off conflicting clean gradients."""
    projected = []
    for clean, robust, should_project in zip(clean_grads, robust_grads, project_mask):
        if robust is None:
            projected.append(None)
            continue
        if clean is None or not should_project:
            projected.append(robust)
            continue
        dot = torch.sum(clean * robust)
        if dot < 0:
            robust = robust - dot / clean.square().sum().clamp_min(1e-12) * clean
        projected.append(robust)
    return [
        None if clean is None and robust is None else
        (robust if clean is None else clean + (robust if robust is not None else 0))
        for clean, robust in zip(clean_grads, projected)
    ]


def corruption_objective(
    corrupt_output,
    clean_target,
    flow_gt,
    valid,
    group_ids,
    *,
    gamma=0.85,
    eta_final_epe=0.05,
    epe_alpha=0.0,
    lambda_r=0.1,
    temperature=0.3,
    use_corrupt_supervision=True,
    use_group_dro=True,
    feature_loss=0.0,
    clean_feature=None,
    corrupt_feature=None,
    use_soft_worst=False,
    soft_beta=0.2,
    soft_temperature=0.3,
    corruption_ema=None,
    corruption_ema_decay=0.9,
):
    corr_nf = sequence_loss_per_sample(corrupt_output, flow_gt, valid, gamma)
    corr_epe = final_epe_per_sample(corrupt_output, flow_gt, valid)
    corr_sequence_epe = sequence_epe_per_sample(corrupt_output, flow_gt, valid, gamma)
    consistency = consistency_per_sample(corrupt_output, clean_target, valid)
    per_sample = lambda_r * consistency
    if use_corrupt_supervision:
        per_sample = per_sample + corr_nf + eta_final_epe * corr_epe + epe_alpha * corr_sequence_epe
    if feature_loss and clean_feature is not None and corrupt_feature is not None:
        per_sample = per_sample + feature_loss * feature_consistency_per_sample(
            clean_feature, corrupt_feature, valid
        )
    groups = {}
    if use_soft_worst:
        objective, corruption_means, effective_temperature = soft_worst_corruption(
            per_sample,
            group_ids,
            beta=soft_beta,
            temperature=soft_temperature,
            ema=corruption_ema,
            ema_decay=corruption_ema_decay,
        )
    elif use_group_dro:
        objective, groups = group_dro(per_sample, group_ids, temperature)
        corruption_means, effective_temperature = {}, None
    else:
        objective, corruption_means, effective_temperature = per_sample.mean(), {}, None
    return {
        "loss": objective,
        "per_sample": per_sample,
        "nf": corr_nf.mean(),
        "epe": corr_epe.mean(),
        "sequence_epe": corr_sequence_epe.mean(),
        "consistency": consistency.mean(),
        "groups": groups,
        "corruption_means": corruption_means,
        "soft_temperature": effective_temperature,
    }
