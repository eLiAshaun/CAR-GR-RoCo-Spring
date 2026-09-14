"""Independent Stage3/Stage4 training entry for WAFT-RX."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

SOURCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_ROOT.parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from config.parser import json_to_args
from criterion.robust_loss import (
    clean_supervised_loss,
    corruption_objective,
    gradient_cosine,
    l2_sp_loss,
    project_robust_gradients,
)
from dataloader.paired_corruptions import CORRUPTIONS, GROUP_NAMES, PairedSpringCorruptionDataset
from model import fetch_model
from utils.utils import load_ckpt


def _get(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_config(path: str | Path) -> SimpleNamespace:
    args = json_to_args(str(path))
    defaults = {
        "algorithm": "waft-rx",
        "feature_encoder": "dav2",
        "iterative_module": "vits",
        "stage": "stage3",
        "iters": 5,
        "var_min": 0,
        "var_max": 10,
        "gamma": 0.85,
        "epsilon": 1e-8,
        "clip": 1.0,
        "precision": "bf16",
        "batch_size": 2,
        "grad_accum_steps": 1,
        "num_workers": min(8, os.cpu_count() or 1),
        "prefetch_factor": 2,
        "persistent_workers": True,
        "num_steps": 10,
        "checkpoint_every": 250,
        "log_every": 10,
        "image_size": [540, 960],
        "scale": -1.0,
        "rx_zira": False,
        "rx_msri": False,
        "rx_msri_s4": False,
        "rx_cri": False,
        "rx_udg": False,
        "train_new_modules": True,
        "train_fusion_heads": True,
        "train_last_refine_blocks": 0,
        "train_dav2_last_blocks": 0,
        "train_dpt_head": False,
        "lr_new": 2e-4,
        "lr_heads": 3e-5,
        "lr_refine": 8e-6,
        "lr_dav2": 1e-6,
        "wdecay_new": 1e-4,
        "wdecay_pretrained": 1e-5,
        "warmup_ratio": 0.05,
        "eta_final_epe": 0.05,
        "sequence_epe_alpha": 0.0,
        "lambda_clean": 1.0,
        "lambda_corrupt": 1.0,
        "lambda_r": 0.1,
        "lambda_sp": 0.0,
        "dro_temperature": 0.3,
        "use_corrupt_supervision": True,
        "use_consistency": True,
        "use_group_dro": True,
        "use_soft_worst": False,
        "soft_beta": 0.2,
        "soft_temperature": 0.3,
        "corruption_ema_decay": 0.9,
        "feature_loss": 0.0,
        "use_cgp": False,
        "use_ema_teacher": False,
        "ema_decay": 0.999,
        "severity_min": 1,
        "severity_max": 4,
        "corruption_names": None,
        "data_split": "train",
        "scene_names": None,
        "dataset_root": "data/spring",
        "restore_ckpt": None,
        "run_dir": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    args.config_path = str(path)
    return args


def _patch_pretrained_constructors():
    """Use local WAFT checkpoint weights; never trigger network downloads."""
    import timm
    import model.waft_a2 as waft_a2

    original_timm = timm.create_model
    original_depth = waft_a2.DepthAnythingFeature

    def create_model(*model_args, **kwargs):
        kwargs["pretrained"] = False
        return original_timm(*model_args, **kwargs)

    def depth_feature(model_name="vits", pretrained=True, lvl=-3):
        return original_depth(model_name, pretrained=False, lvl=lvl)

    timm.create_model = create_model
    waft_a2.DepthAnythingFeature = depth_feature
    return timm, original_timm, waft_a2, original_depth


def build_model(args):
    timm, original_timm, waft_a2, original_depth = _patch_pretrained_constructors()
    try:
        return fetch_model(args)
    finally:
        timm.create_model = original_timm
        waft_a2.DepthAnythingFeature = original_depth


def _state_dict(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def load_parent_checkpoint(model, checkpoint: str | Path, allow_new: bool = False):
    state = _state_dict(checkpoint)
    if not allow_new:
        result = model.load_state_dict(state, strict=True)
        return result.missing_keys, result.unexpected_keys
    result = model.load_state_dict(state, strict=False)
    expected = {key for key in model.state_dict() if key not in state}
    missing = set(result.missing_keys)
    if missing != expected or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint compatibility mismatch: missing={sorted(missing - expected)} "
            f"unexpected={result.unexpected_keys}"
        )
    return sorted(missing), result.unexpected_keys


def _block_index(name: str, prefix: str):
    match = re.match(re.escape(prefix) + r"(\d+)\.", name)
    return int(match.group(1)) if match else None


def configure_trainable(model, args):
    for parameter in model.parameters():
        parameter.requires_grad = False

    categories = {}
    new_prefixes = []
    for flag, prefix in (
        ("rx_zira", "zira."),
        ("rx_msri", "msri."),
        ("rx_cri", "cri."),
        ("rx_udg", "udg."),
    ):
        if getattr(args, flag, False):
            new_prefixes.append(prefix)

    refine_prefix = "refine_net.blks."
    dav2_prefix = "encoder.encoder.blocks."
    block_count = len(getattr(getattr(model, "refine_net", None), "blks", []))
    dav2_block_count = len(getattr(getattr(getattr(model, "encoder", None), "encoder", None), "blocks", []))

    def category(name):
        if getattr(args, "train_new_modules", True) and any(name.startswith(p) for p in new_prefixes):
            return "new"
        heads = (
            "fmap_conv.",
            "hidden_conv.",
            "warp_linear.",
            "refine_transform.",
            "flow_head.",
            "upsample_weight.",
        )
        if getattr(args, "train_fusion_heads", True) and name.startswith(heads):
            return "heads"
        index = _block_index(name, refine_prefix)
        if index is not None and index >= block_count - int(getattr(args, "train_last_refine_blocks", 0)):
            return "refine"
        index = _block_index(name, dav2_prefix)
        if index is not None and index >= dav2_block_count - int(getattr(args, "train_dav2_last_blocks", 0)):
            return "dav2"
        if getattr(args, "train_dpt_head", False) and name.startswith("encoder.dpt_head."):
            return "dav2"
        return None

    for name, parameter in model.named_parameters():
        group = category(name)
        if group:
            parameter.requires_grad = True
            categories[name] = group

    learning_rates = {
        "new": float(_get(args, "lr_new", 2e-4)),
        "heads": float(_get(args, "lr_heads", 3e-5)),
        "refine": float(_get(args, "lr_refine", 8e-6)),
        "dav2": float(_get(args, "lr_dav2", 1e-6)),
    }
    weight_decays = {
        "new": float(_get(args, "wdecay_new", 1e-4)),
        "heads": float(_get(args, "wdecay_pretrained", 1e-5)),
        "refine": float(_get(args, "wdecay_pretrained", 1e-5)),
        "dav2": float(_get(args, "wdecay_pretrained", 1e-5)),
    }
    groups = []
    for group_name in ("new", "heads", "refine", "dav2"):
        parameters = [p for name, p in model.named_parameters() if categories.get(name) == group_name]
        if parameters:
            groups.append(
                {
                    "name": group_name,
                    "params": parameters,
                    "lr": learning_rates[group_name],
                    "weight_decay": weight_decays[group_name],
                }
            )
    assigned = [id(p) for group in groups for p in group["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    if sorted(assigned) != sorted(trainable) or len(assigned) != len(set(assigned)):
        raise RuntimeError("trainable parameters are not in exactly one optimizer group")
    if not groups:
        raise RuntimeError("configuration leaves no trainable parameters")
    audit = {
        "groups": {
            group["name"]: {
                "parameter_count": sum(p.numel() for p in group["params"]),
                "tensor_count": len(group["params"]),
                "names": [name for name, value in categories.items() if value == group["name"]],
            }
            for group in groups
        },
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    return groups, audit, categories


def _set_train_mode(model, args):
    model.train()
    model.fnet.eval()
    model.encoder.eval()
    if getattr(args, "train_dpt_head", False):
        model.encoder.dpt_head.train()
    if not any(name.startswith("refine_net.") and p.requires_grad for name, p in model.named_parameters()):
        model.refine_net.eval()


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device, precision):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _data_loader(args):
    crop_size = tuple(int(value) for value in args.image_size)
    scale = float(_get(args, "scale", -1.0))
    names = getattr(args, "corruption_names", None)
    if isinstance(names, str):
        names = tuple(value for value in names.split(",") if value)
    dataset = PairedSpringCorruptionDataset(
        root=resolve_path(_get(args, "dataset_root", "data/spring")),
        split=_get(args, "data_split", "train"),
        crop_size=crop_size,
        min_scale=scale,
        max_scale=scale + 0.2,
        do_flip=bool(_get(args, "do_flip", True)),
        paired=bool(_get(args, "paired", _get(args, "stage", "stage3") == "stage4")),
        scene_names=_get(args, "scene_names", None),
        corruption_names=names,
        severity_min=int(_get(args, "severity_min", 1)),
        severity_max=int(_get(args, "severity_max", 4)),
        seed=int(_get(args, "seed", 42)),
    )
    workers = max(0, int(_get(args, "num_workers", 0)))
    options = {
        "batch_size": int(args.batch_size),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if workers:
        options["persistent_workers"] = bool(_get(args, "persistent_workers", True))
        options["prefetch_factor"] = max(1, int(_get(args, "prefetch_factor", 2)))
    return dataset, DataLoader(dataset, **options)


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu()) if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _run_dir(args):
    if getattr(args, "run_dir", None):
        return resolve_path(args.run_dir)
    root = REPO_ROOT / "experiments" / ("stage4" if args.stage == "stage4" else "stage3")
    return root / str(args.name)


def _write_run_metadata(run_dir, args, audit, checkpoint):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "config.resolved.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n")
    (run_dir / "trainable_parameters.txt").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    (run_dir / "environment.txt").write_text(
        "\n".join(
            [
                f"python={sys.version}",
                f"platform={platform.platform()}",
                f"torch={torch.__version__}",
                f"torch_cuda={torch.version.cuda}",
                f"gpu={gpu}",
                f"parent_checkpoint={checkpoint}",
                "version_control=not_collected_by_request",
            ]
        )
        + "\n"
    )
    manifest = {
        "run_id": str(args.name),
        "stage": args.stage,
        "parent_checkpoint": str(checkpoint),
        "split_id": _get(args, "split_id", "spring_train"),
        "seed": int(_get(args, "seed", 42)),
        "config": vars(args),
        "audit": audit,
        "corruption_generator": "paired_corruptions.v1; supervised_corruptions=19; elastic_transform_excluded_until_flow_jacobian",
        "sealed_evaluation": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def _save_checkpoint(path, model, optimizer, scheduler, step, audit, teacher=None):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "audit": audit,
    }
    if teacher is not None:
        payload["ema_model"] = teacher.state_dict()
    torch.save(payload, path)


def _update_ema(teacher, student, decay):
    with torch.no_grad():
        for target, source in zip(teacher.parameters(), student.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)
        for target, source in zip(teacher.buffers(), student.buffers()):
            target.copy_(source)


def load_training_checkpoint(path, model, optimizer=None, scheduler=None, teacher=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if teacher is not None and "ema_model" in payload:
        teacher.load_state_dict(payload["ema_model"], strict=True)
    return int(payload["step"])


def _scheduler(optimizer, steps, warmup_ratio):
    warmup = max(1, round(steps * float(warmup_ratio)))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_training(args):
    _seed_everything(int(_get(args, "seed", 42)))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    checkpoint = resolve_path(args.restore_ckpt)
    model = build_model(args)
    load_parent_checkpoint(model, checkpoint, allow_new=args.algorithm == "waft-rx")
    groups, audit, categories = configure_trainable(model, args)
    model.to(device)
    _set_train_mode(model, args)
    teacher = None
    if bool(_get(args, "use_ema_teacher", False)):
        teacher = copy.deepcopy(model).to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(groups, eps=float(_get(args, "epsilon", 1e-8)))
    steps = int(args.num_steps)
    scheduler = _scheduler(optimizer, steps, _get(args, "warmup_ratio", 0.05))
    anchors = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and categories.get(name) != "new" and float(_get(args, "lambda_sp", 0.0))
    }
    run_dir = _run_dir(args)
    _write_run_metadata(run_dir, args, audit, checkpoint)
    dataset, loader = _data_loader(args)
    iterator = iter(loader)
    accumulation = max(1, int(_get(args, "grad_accum_steps", 1)))
    corruption_ema = torch.zeros(len(CORRUPTIONS), device=device)
    optimizer.zero_grad(set_to_none=True)
    log_path = run_dir / "train_log.jsonl"
    started = time.perf_counter()

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        clean1 = batch["clean_image1"].to(device, non_blocking=True)
        clean2 = batch["clean_image2"].to(device, non_blocking=True)
        corrupt1 = batch["corrupt_image1"].to(device, non_blocking=True)
        corrupt2 = batch["corrupt_image2"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        group_ids = batch["group_id"].to(device, non_blocking=True)
        corruption_ids = batch["corruption_id"].to(device, non_blocking=True)
        if args.stage == "stage4":
            with _autocast(device, args.precision):
                clean_output = model(clean1, clean2, flow_gt=flow)
                clean_terms = clean_supervised_loss(
                    clean_output,
                    flow,
                    valid,
                    args.gamma,
                    args.eta_final_epe,
                    _get(args, "sequence_epe_alpha", 0.0),
                )
            clean_loss = clean_terms["per_sample"].mean()
            trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if args.use_cgp:
                clean_grads = torch.autograd.grad(
                    clean_loss, trainable_parameters, allow_unused=True
                )
            else:
                (_get(args, "lambda_clean", 1.0) * clean_loss / accumulation).backward()
                clean_grads = None
            if teacher is None:
                clean_target = clean_output["flow"][-1].detach()
            else:
                with torch.no_grad(), _autocast(device, args.precision):
                    clean_target = teacher(clean1, clean2)["flow"][-1].detach()
            clean_feature = clean_output.get("feature", [None])[0]
            clean_nf, clean_epe = clean_terms["nf"], clean_terms["epe"]
            del clean_output, clean_terms
            with _autocast(device, args.precision):
                corrupt_output = model(corrupt1, corrupt2, flow_gt=flow)
                robust_terms = corruption_objective(
                    corrupt_output,
                    clean_target,
                    flow,
                    valid,
                    corruption_ids if _get(args, "use_soft_worst", False) else group_ids,
                    gamma=args.gamma,
                    eta_final_epe=args.eta_final_epe,
                    epe_alpha=_get(args, "sequence_epe_alpha", 0.0),
                    lambda_r=args.lambda_r if args.use_consistency else 0.0,
                    temperature=args.dro_temperature,
                    use_corrupt_supervision=args.use_corrupt_supervision,
                    use_group_dro=args.use_group_dro,
                    feature_loss=args.feature_loss,
                    clean_feature=clean_feature,
                    corrupt_feature=corrupt_output.get("feature", [None])[0],
                    use_soft_worst=_get(args, "use_soft_worst", False),
                    soft_beta=_get(args, "soft_beta", 0.2),
                    soft_temperature=_get(args, "soft_temperature", 0.3),
                    corruption_ema=corruption_ema,
                    corruption_ema_decay=_get(args, "corruption_ema_decay", 0.9),
                )
                sp = l2_sp_loss(model, anchors) if args.lambda_sp else robust_terms["loss"].new_zeros(())
                total_loss = robust_terms["loss"] + args.lambda_sp * sp
            if args.use_cgp:
                robust_grads = torch.autograd.grad(
                    total_loss, trainable_parameters, allow_unused=True
                )
                project_mask = [
                    categories.get(name) in {"new", "heads"}
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                combined_grads = project_robust_gradients(clean_grads, robust_grads, project_mask)
                for parameter, gradient in zip(trainable_parameters, combined_grads):
                    if gradient is not None:
                        if parameter.grad is None:
                            parameter.grad = gradient / accumulation
                        else:
                            parameter.grad.add_(gradient / accumulation)
                cosine = gradient_cosine(clean_grads, robust_grads)
            else:
                (_get(args, "lambda_corrupt", 1.0) * total_loss / accumulation).backward()
                cosine = None
            metrics = {
                "clean_nf": clean_nf,
                "clean_epe": clean_epe,
                "corr_nf": robust_terms["nf"],
                "corr_epe": robust_terms["epe"],
                "corr_sequence_epe": robust_terms["sequence_epe"],
                "consistency": robust_terms["consistency"],
                "dro": robust_terms["loss"],
                "l2_sp": sp,
                "soft_temperature": robust_terms["soft_temperature"],
            }
            if cosine is not None:
                metrics["gradient_cosine"] = cosine
            del corrupt_output, robust_terms
        else:
            with _autocast(device, args.precision):
                output = model(clean1, clean2, flow_gt=flow)
                terms = clean_supervised_loss(
                    output,
                    flow,
                    valid,
                    args.gamma,
                    args.eta_final_epe,
                    _get(args, "sequence_epe_alpha", 0.0),
                )
                total_loss = _get(args, "lambda_clean", 1.0) * terms["per_sample"].mean()
            (total_loss / accumulation).backward()
            metrics = {
                "loss": total_loss,
                "nf": terms["nf"],
                "epe": terms["epe"],
                "sequence_epe": terms["sequence_epe"],
            }
            del output, terms

        if (step + 1) % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None:
                _update_ema(teacher, model, float(args.ema_decay))
        if step % max(1, int(args.log_every)) == 0 or step == steps - 1:
            record = {
                "step": step + 1,
                "elapsed_s": time.perf_counter() - started,
                "lr": [group["lr"] for group in optimizer.param_groups],
                **{key: _json_safe(value) for key, value in metrics.items() if not callable(value)},
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if (step + 1) % max(1, int(args.checkpoint_every)) == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"step_{step + 1:07d}.pt",
                model,
                optimizer,
                scheduler,
                step + 1,
                audit,
                teacher,
            )

    final = run_dir / "checkpoints" / "final.pt"
    _save_checkpoint(final, model, optimizer, scheduler, steps, audit, teacher)
    (run_dir / "decision.md").write_text(
        "# Stage34 run\n\n"
        f"Completed `{steps}` steps. Metrics are recorded for comparison; no automatic promotion gate was applied.\n"
    )
    return {"run_dir": str(run_dir), "checkpoint": str(final), "audit": audit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--run-dir")
    parser.add_argument("--smoke", action="store_true")
    cli = parser.parse_args()
    args = load_config(cli.cfg)
    for key, value in (
        ("num_steps", cli.steps),
        ("batch_size", cli.batch_size),
        ("num_workers", cli.workers),
        ("device", cli.device),
        ("precision", cli.precision),
        ("run_dir", cli.run_dir),
    ):
        if value is not None:
            setattr(args, key, value)
    if cli.smoke:
        args.num_steps = 10
        args.batch_size = 1
        args.num_workers = 0
        args.image_size = [224, 336]
        args.scale = 0
        args.persistent_workers = False
    run_training(args)


if __name__ == "__main__":
    main()
