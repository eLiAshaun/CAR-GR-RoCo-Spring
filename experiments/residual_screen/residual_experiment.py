"""Short, isolated WAFT-DAv2-A2 residual screen.

The parent model is loaded from the existing external-benchmark adapter and is
never optimized.  This file owns only the residual heads, fixed split, proxy
corruptions, measurements, and run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments" / "external_benchmark" / "sources" / "WAFT"
OUT = ROOT / "results" / "residual_screen"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from experiments.external_benchmark import adapters, local_protocol


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGE1 = _load_module(ROOT / "experiments/stage1_tournament/stage1.py", "residual_stage1")
FAST = _load_module(
    ROOT / "experiments/stage2_5/fast_benchmark/benchmark.py", "residual_fast_benchmark"
)
WAFT_UTILS = _load_module(SOURCE / "utils/utils.py", "residual_waft_utils")

PARENT_MODEL = "waft_dav2_a2"
PARENT_CHECKPOINT = ROOT / "experiments/external_benchmark/weights/waft/extracted/waft_dav2_a2.pth"
PARENT_ARCHIVE = ROOT / "experiments/external_benchmark/weights/waft/a2.zip"
CONFIG = ROOT / "experiments/external_benchmark/sources/WAFT/config/a2/dav2/tar-c-t-spring-540p.json"
TRAIN_SCENES = ROOT / "experiments/stage2_5/train_scenes.txt"
DEV_SCENES = ROOT / "experiments/stage2_5/dev_scenes.txt"
CONFIRM_SCENES = ROOT / "experiments/stage2_5/confirm_scenes.txt"
CONTRACT_ID = "waft-r1-r8-v2"
EXCLUDED_CORRUPTIONS = {"elastic_transform", "glass_blur"}

GROUPS = {
    "color": ("brightness", "contrast", "saturate"),
    "blur": ("defocus_blur", "gaussian_blur", "motion_blur", "zoom_blur"),
    "noise": ("gaussian_noise", "impulse_noise", "shot_noise", "speckle_noise"),
    "quality": ("jpeg_compression", "pixelate"),
    "weather": ("fog", "frost", "rain", "snow", "spatter"),
}
GROUP_FOR = {condition: group for group, conditions in GROUPS.items() for condition in conditions}
TRAIN_CORRUPTIONS = tuple(name for name in FAST.CORRUPTIONS if name not in EXCLUDED_CORRUPTIONS)
ALL_CONDITIONS = ("clean", *TRAIN_CORRUPTIONS)
VARIANTS = ("R1", "R2", "R3", "R4")
RDIAG_BUCKETS = (
    "clean", "corrupt", *GROUPS,
    "motion_0-10", "motion_10-40", "motion_40+",
    "error_<0.25", "error_0.25-0.5", "error_0.5-1", "error_1-3", "error_>3",
)
_GT_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_CONDITION_IMAGE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temp.replace(path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _write_block_review(name: str, checks: dict[str, bool]) -> None:
    passed = bool(checks) and all(checks.values())
    _atomic_json(OUT / "block_reviews" / f"{name}.json", {"contract_id": CONTRACT_ID, "design": "experiments/waft_r1_r8_design.md", "passed": passed, "checks": checks, "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if not passed:
        raise RuntimeError(f"{name} document review failed: {checks}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_lines(path: Path) -> tuple[str, ...]:
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def _split_rows(path: Path) -> list[dict]:
    return local_protocol.build_labeled_manifest(_read_lines(path), local_protocol.DATA_ROOT)


def _split_hash(path: Path, rows: list[dict]) -> str:
    payload = {
        "scene_file": str(path),
        "scene_file_sha256": _sha256(path),
        "manifest": rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _gpu_query() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 4:
            rows.append({"index": values[0], "name": values[1], "memory_total": values[2], "driver": values[3]})
    return rows


def build_registry(eval_count: int) -> dict:
    train_rows = _split_rows(TRAIN_SCENES)
    dev_rows = _split_rows(DEV_SCENES)
    confirm_rows = _split_rows(CONFIRM_SCENES)
    if set(row["scene"] for row in train_rows) & set(row["scene"] for row in dev_rows + confirm_rows):
        raise RuntimeError("train/dev/confirm scenes overlap")
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repository": {
            "root": str(ROOT),
            "git_commit": "not-a-git-repository",
            "waft_source_commit": "b152ff1cad1af8c185ee7b141997c48ff3334c87",
            "waft_source_dirty": True,
        },
        "parent": {
            "model_id": PARENT_MODEL,
            "checkpoint": str(PARENT_CHECKPOINT),
            "checkpoint_sha256": _sha256(PARENT_CHECKPOINT),
            "archive": str(PARENT_ARCHIVE),
            "archive_sha256": _sha256(PARENT_ARCHIVE),
            "config": str(CONFIG),
            "config_sha256": _sha256(CONFIG),
            "iters": 5,
            "backbone_frozen": True,
            "return_aux_default": False,
        },
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "gpu_count": torch.cuda.device_count(),
            "gpus": _gpu_query(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "precision": "bf16",
        },
        "data": {
            "spring": str(ROOT / "data/spring"),
            "robust_spring": str(ROOT / "data/robust_spring"),
            "train_scene_file": str(TRAIN_SCENES),
            "dev_scene_file": str(DEV_SCENES),
            "confirm_scene_file": str(CONFIRM_SCENES),
            "train_manifest_count": len(train_rows),
            "dev_manifest_count": len(dev_rows),
            "confirm_manifest_count": len(confirm_rows),
            "train_split_hash": _split_hash(TRAIN_SCENES, train_rows),
            "dev_split_hash": _split_hash(DEV_SCENES, dev_rows),
            "confirm_split_hash": _split_hash(CONFIRM_SCENES, confirm_rows),
        },
        "protocol": {
            "contract_id": CONTRACT_ID,
            "eval_sample_count": eval_count,
            "eval_split": "stage2_5/dev_scenes.txt",
            "conditions": list(ALL_CONDITIONS),
            "train_clean_fraction": 0.5,
            "train_corruption_severity": 0.75,
            "excluded_geometric_corruptions": sorted(EXCLUDED_CORRUPTIONS),
            "train_corruption_seed_namespace": "residual-train-v2",
            "eval_corruption_seed_namespace": "residual-eval-v2",
            "robustspring_gt": "private/unavailable",
            "real_robustspring_track": "clean-to-corrupt stability only",
            "steps": 2500,
            "eval_every": 500,
        },
    }


def _read_bgr(path: str | Path) -> np.ndarray:
    image = cv.imread(str(path), cv.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read {path}")
    return image


def _read_eval_gt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    key = str(path)
    if key not in _GT_CACHE:
        _GT_CACHE[key] = STAGE1.read_spring_gt(Path(path))
    return _GT_CACHE[key]


def _resize_bgr(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv.resize(image, (width, height), interpolation=cv.INTER_LINEAR)


def _resize_flow(flow: np.ndarray, height: int, width: int) -> np.ndarray:
    old_height, old_width = flow.shape[:2]
    result = cv.resize(flow, (width, height), interpolation=cv.INTER_LINEAR)
    result[..., 0] *= width / old_width
    result[..., 1] *= height / old_height
    return result.astype(np.float32, copy=False)


def _resize_valid(valid: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv.resize(valid.astype(np.uint8), (width, height), interpolation=cv.INTER_NEAREST).astype(bool)


class SpringResidualDataset(Dataset):
    def __init__(self, rows: list[dict], length: int, height: int, width: int, seed: int):
        if not rows:
            raise ValueError("training split produced no labeled rows")
        if length < 0:
            raise ValueError(f"dataset length must be non-negative, got {length}")
        self.rows = rows
        self.length = length
        self.height = height
        self.width = width
        self.seed = seed
        self.offset = 0
        stream_rng = np.random.default_rng(seed + 991)
        self.row_stream = []
        while len(self.row_stream) < length:
            self.row_stream.extend(stream_rng.permutation(len(rows)).tolist())
        self.row_stream = self.row_stream[:length]

    def __len__(self) -> int:
        return self.length

    def skip(self, count: int) -> None:
        if not 0 <= count <= self.length:
            raise ValueError(f"invalid dataset skip: {count}/{self.length}")
        self.row_stream = self.row_stream[count:]
        self.length -= count
        self.offset += count

    def __getitem__(self, index: int):
        row = self.rows[self.row_stream[index]]
        image1 = _read_bgr(row["image1"])
        image2 = _read_bgr(row["image2"])
        ground_truth, valid = STAGE1.read_spring_gt(Path(row["gt"]))
        rng = np.random.default_rng(self.seed + 1000003 * (self.offset + index + 1))
        if rng.random() < 0.5:
            corruption = TRAIN_CORRUPTIONS[int(rng.integers(len(TRAIN_CORRUPTIONS)))]
            seed = int(rng.integers(0, 2**31 - 1))
            corrupted1, corrupted2 = FAST.corrupt_pair(image1, image2, corruption, seed)
            image1 = np.clip(image1.astype(np.float32) + 0.75 * (corrupted1.astype(np.float32) - image1), 0, 255).astype(np.uint8)
            image2 = np.clip(image2.astype(np.float32) + 0.75 * (corrupted2.astype(np.float32) - image2), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            image1 = image1[:, ::-1].copy()
            image2 = image2[:, ::-1].copy()
            ground_truth = ground_truth[:, ::-1].copy()
            ground_truth[..., 0] *= -1
            valid = valid[:, ::-1].copy()
        image1 = _resize_bgr(image1, self.height, self.width)[:, :, ::-1].copy()
        image2 = _resize_bgr(image2, self.height, self.width)[:, :, ::-1].copy()
        ground_truth = _resize_flow(ground_truth, self.height, self.width)
        valid = _resize_valid(valid, self.height, self.width)
        return (
            torch.from_numpy(image1).permute(2, 0, 1).float(),
            torch.from_numpy(image2).permute(2, 0, 1).float(),
            torch.from_numpy(ground_truth).permute(2, 0, 1).float(),
            torch.from_numpy(valid.copy()),
        )

    def __getitems__(self, indices: list[int]):
        # Keep worker=0 while overlapping the four independent NFS/HDF5 reads.
        if not indices:
            return []
        with ThreadPoolExecutor(max_workers=min(4, len(indices))) as pool:
            return list(pool.map(self.__getitem__, indices))


def _amp(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _make_head(variant: str) -> nn.Module:
    return {
        "R1": DeterministicResidual(hidden=64),
        "R2": RiskGatedResidual(hidden=64),
        "R3": LowResolutionResidual(hidden=32),
        "R4": ResidualODE(hidden=32, steps=3),
    }[variant]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SmallResidualNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int, blocks: int = 2):
        super().__init__()
        self.project = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden) for _ in range(blocks)))
        self.output = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(F.gelu(self.project(x))))


class GateNet(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(67, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class DeterministicResidual(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = SmallResidualNet(67, hidden)

    def forward(self, bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"delta": self.net(bundle["features"])}


class RiskGatedResidual(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = SmallResidualNet(67, hidden)
        self.gate = GateNet()

    def forward(self, bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        delta = self.net(bundle["features"])
        gate = self.gate(bundle["gate_features"])
        return {"delta": gate * delta, "raw_delta": delta, "gate": gate}


class LowResolutionResidual(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = SmallResidualNet(67, hidden)

    def forward(self, bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        height, width = bundle["f0"].shape[-2:]
        size = (max(1, height // 8), max(1, width // 8))
        low = F.interpolate(bundle["features"], size=size, mode="bilinear", align_corners=False)
        low = low.clone()
        low[:, 64:66] /= 8.0
        delta_low = self.net(low)
        delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False) * 8.0
        return {"delta": delta, "delta_low": delta_low}


class ResidualODE(nn.Module):
    def __init__(self, hidden: int, steps: int):
        super().__init__()
        self.steps = steps
        self.context = nn.Conv2d(67, hidden, 3, padding=1)
        field_channels = 2 + hidden + 2 + 1 + 4
        self.field = nn.Sequential(
            nn.Conv2d(field_channels, hidden, 3, padding=1),
            nn.GELU(),
            ResidualBlock(hidden),
            ResidualBlock(hidden),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)
        self.gate = GateNet()

    @staticmethod
    def _time(t: float, reference: torch.Tensor) -> torch.Tensor:
        values = (math.sin(math.pi * t), math.cos(math.pi * t), math.sin(2 * math.pi * t), math.cos(2 * math.pi * t))
        return reference.new_tensor(values).view(1, 4, 1, 1).expand(reference.shape[0], -1, reference.shape[-2], reference.shape[-1])

    def _field(self, residual: torch.Tensor, context: torch.Tensor, bundle: dict[str, torch.Tensor], t: float) -> torch.Tensor:
        return self.field(torch.cat([residual, context, bundle["features"][:, 64:66], bundle["features"][:, -1:], self._time(t, residual)], dim=1))

    def forward(self, bundle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = F.gelu(self.context(bundle["features"]))
        residual = torch.zeros_like(bundle["f0"])
        states = []
        dt = 1.0 / self.steps
        for index in range(self.steps):
            t = index * dt
            first = self._field(residual, context, bundle, t)
            midpoint = residual + 0.5 * dt * first
            second = self._field(midpoint, context, bundle, t + 0.5 * dt)
            residual = residual + dt * second
            states.append(residual)
        gate = self.gate(bundle["gate_features"])
        return {"delta": gate * residual, "raw_delta": residual, "gate": gate, "states": states}


def _forward_native_bundle(model: nn.Module, image1: torch.Tensor, image2: torch.Tensor, device: torch.device, amp_dtype: torch.dtype | None, include_trajectory: bool = False) -> dict[str, torch.Tensor]:
    with torch.no_grad(), _amp(device, amp_dtype):
        output = model(image1, image2) if getattr(model, "_residual_tuple_output", False) else model(image1, image2, return_aux=True)
    if isinstance(output, tuple):
        flow, fmap1, fmap2, hidden_raw, flow_low, last_update = output
        output = {"flow": [flow], "aux": {"fmap1": fmap1, "fmap2": fmap2, "hidden": hidden_raw, "flow_low": flow_low, "last_update": last_update}}
    aux = output["aux"]
    fmap1 = aux["fmap1"].float()
    fmap2 = aux["fmap2"].float()
    flow_low = aux["flow_low"].float()
    coords = WAFT_UTILS.coords_grid(flow_low.shape[0], flow_low.shape[-2], flow_low.shape[-1], flow_low.device) + flow_low
    warped = WAFT_UTILS.bilinear_sampler(fmap2, coords.permute(0, 2, 3, 1))
    q_low = (F.normalize(fmap1, dim=1, eps=1e-6) * F.normalize(warped, dim=1, eps=1e-6)).sum(dim=1, keepdim=True)
    result = {
        "f0": output["flow"][-1].float(),
        "hidden_low": aux["hidden"].float(),
        "q_low": q_low.float(),
        "last_update_low": aux["last_update"].float(),
    }
    if include_trajectory and "trajectory_low" in aux:
        result["trajectory_low"] = torch.stack([value.float() for value in aux["trajectory_low"]])
    return result


def _expand_native_bundle(native: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    f0 = native["f0"].float()
    size = f0.shape[-2:]
    hidden = F.interpolate(native["hidden_low"].float(), size=size, mode="bilinear", align_corners=True)
    q = F.interpolate(native["q_low"].float(), size=size, mode="bilinear", align_corners=True)
    last_update = F.interpolate(native["last_update_low"].float(), size=size, mode="bilinear", align_corners=True) * 2.0
    # Keep the frozen model's native features, but present the trainable head
    # with bounded scales.  f0/updates are pixel units; unscaled rare large
    # flows can overflow a conv backward even when the final EPE is finite.
    hidden = torch.nan_to_num(hidden, nan=0.0, posinf=32.0, neginf=-32.0).clamp(-32.0, 32.0)
    f0_head = torch.nan_to_num(f0, nan=0.0, posinf=4096.0, neginf=-4096.0).clamp(-4096.0, 4096.0)
    q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    last_update = torch.nan_to_num(last_update, nan=0.0, posinf=4096.0, neginf=-4096.0).clamp(-4096.0, 4096.0)
    features = torch.cat([hidden / 8.0, f0_head / 32.0, q], dim=1)
    gate_features = torch.cat([hidden / 8.0, q, last_update.norm(dim=1, keepdim=True) / 32.0, f0_head.norm(dim=1, keepdim=True) / 32.0], dim=1)
    result = {"features": features, "gate_features": gate_features, "f0": f0}
    if "trajectory_low" in native:
        result["trajectory"] = [F.interpolate(flow.float() * 2.0, size=size, mode="bilinear", align_corners=True) for flow in native["trajectory_low"]]
    return result


def _forward_bundle(model: nn.Module, image1: torch.Tensor, image2: torch.Tensor, device: torch.device, amp_dtype: torch.dtype | None, include_trajectory: bool = False) -> dict[str, torch.Tensor]:
    return _expand_native_bundle(_forward_native_bundle(model, image1, image2, device, amp_dtype, include_trajectory))


def _valid_mask(prediction: torch.Tensor, ground_truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(ground_truth, dim=1)
    return valid.bool() & torch.isfinite(ground_truth).all(dim=1) & torch.isfinite(prediction).all(dim=1) & (magnitude < 2500)


def _flow_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = _valid_mask(prediction, ground_truth, valid)
    if not mask.any():
        return torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    diff = torch.nan_to_num(prediction.float(), nan=0.0, posinf=4096.0, neginf=-4096.0)
    diff = diff - torch.nan_to_num(ground_truth.float(), nan=0.0, posinf=4096.0, neginf=-4096.0)
    diff = diff.clamp(-8192.0, 8192.0)
    value = torch.sqrt((diff * diff).sum(dim=1) + 1e-6)
    return value[mask].mean()


def _loss(head: nn.Module, bundle: dict[str, torch.Tensor], ground_truth: torch.Tensor, valid: torch.Tensor, output: dict[str, torch.Tensor] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if output is None:
        output = head(bundle)
    prediction = bundle["f0"] + output["delta"]
    loss = _flow_loss(prediction, ground_truth, valid)
    if "gate" in output:
        gate_mask = _valid_mask(bundle["f0"], ground_truth, valid).unsqueeze(1)
        baseline_error = torch.linalg.vector_norm(torch.nan_to_num(bundle["f0"] - ground_truth), dim=1, keepdim=True)
        target = torch.sigmoid((baseline_error - 0.5) / 0.15)
        gate = output["gate"].clamp(1e-5, 1 - 1e-5)
        bce = F.binary_cross_entropy(gate.float(), target.float(), reduction="none")
        if gate_mask.any():
            loss = loss + 0.1 * bce[gate_mask].mean()
        loss = loss + 0.02 * gate.mean() + 0.02 * output["delta"].abs().mean()
    else:
        loss = loss + 0.05 * output["delta"].abs().mean()
    return loss, output


def _step(model: nn.Module, head: nn.Module, optimizer: torch.optim.Optimizer, batch, device: torch.device, amp_dtype: torch.dtype | None) -> tuple[float, dict[str, float]]:
    image1, image2, ground_truth, valid = [value.to(device, non_blocking=True) for value in batch]
    optimizer.zero_grad(set_to_none=True)
    bundle = _forward_bundle(model, image1, image2, device, amp_dtype)
    loss, _ = _loss(head, bundle, ground_truth, valid)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite residual loss: {float(loss.detach().cpu())}")
    loss.backward()
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
        raise FloatingPointError("non-finite residual gradient")
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().float().cpu()), {"batch": float(image1.shape[0])}


class Watchdog:
    def __init__(self, path: Path, variant: str, state: dict):
        self.path = path
        self.variant = variant
        self.state = state
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("timestamp,variant,gpu,utilization_gpu,memory_used_mb,power_w,temperature_c,step,step_time_s,data_time_s,forward_time_s,backward_time_s\n")
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop.wait(10):
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                continue
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            with self.path.open("a") as handle:
                for line in output.splitlines():
                    values = [item.strip() for item in line.split(",")]
                    if len(values) != 5:
                        continue
                    state = ",".join(str(self.state.get(key, "")) for key in ("step", "step_time_s", "data_time_s", "forward_time_s", "backward_time_s"))
                    handle.write(f"{now},{self.variant},{values[0]},{values[1]},{values[2]},{values[3]},{values[4]},{state}\n")


class Reservoir:
    def __init__(self, limit: int = 200000, seed: int = 0):
        self.limit = limit
        self.rng = np.random.default_rng(seed)
        self.values: list[float] = []
        self.seen = 0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if not flat.size:
            return
        self.seen += int(flat.size)
        if len(self.values) < self.limit:
            need = self.limit - len(self.values)
            selected = flat if flat.size <= need else flat[self.rng.choice(flat.size, need, replace=False)]
            self.values.extend(selected.astype(float).tolist())
            flat = flat[need:] if flat.size > need else flat[:0]
        if flat.size and len(self.values) == self.limit:
            positions = self.rng.integers(0, self.limit, size=min(flat.size, self.limit))
            selected = flat[self.rng.choice(flat.size, len(positions), replace=False)] if flat.size > len(positions) else flat
            reservoir = np.asarray(self.values, dtype=np.float64)
            reservoir[np.asarray(positions)] = selected
            self.values = reservoir.tolist()

    def percentile(self, value: float) -> float:
        return float(np.percentile(self.values, value)) if self.values else math.nan


class MetricAccumulator:
    def __init__(self):
        self.statistics: list[dict] = []
        self.motion: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.rescue = [0.0, 0.0]
        self.harm = [0.0, 0.0]
        self.net = {"all": [0.0, 0.0], "baseline_correct": [0.0, 0.0], "baseline_bad": [0.0, 0.0]}
        self.baseline_bins = defaultdict(lambda: [0.0, 0.0])

    def add(self, prediction: np.ndarray, baseline: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray, condition: str, scene: str) -> None:
        self.statistics.append({"condition": condition, "scene": scene, "statistics": local_protocol.accuracy_sufficient_statistics(prediction, ground_truth, valid)})
        error = np.linalg.norm(prediction - ground_truth, axis=-1)
        base_error = np.linalg.norm(baseline - ground_truth, axis=-1)
        motion = np.linalg.norm(ground_truth, axis=-1)
        mask = valid & np.isfinite(error) & np.isfinite(base_error)
        for name, selector in (("<0.25", base_error < 0.25), ("0.25-0.5", (base_error >= 0.25) & (base_error < 0.5)), ("0.5-1", (base_error >= 0.5) & (base_error < 1)), ("1-3", (base_error >= 1) & (base_error < 3)), ("3-10", (base_error >= 3) & (base_error < 10)), (">10", base_error >= 10)):
            selected = mask & selector
            self.baseline_bins[name][0] += float(error[selected].sum())
            self.baseline_bins[name][1] += float(selected.sum())
        for name, selector in (("s0-10", motion < 10), ("s10-40", (motion >= 10) & (motion < 40)), ("s40+", motion >= 40)):
            selected = mask & selector
            self.motion[name][0] += float(error[selected].sum())
            self.motion[name][1] += float(selected.sum())
        if condition == "clean":
            return
        bad = mask & (base_error > 1)
        correct = mask & (base_error < 0.5)
        self.rescue[0] += float(((base_error - error >= 0.5) & bad).sum())
        self.rescue[1] += float(bad.sum())
        self.harm[0] += float(((error - base_error > 0.25) & correct).sum())
        self.harm[1] += float(correct.sum())
        delta = base_error - error
        for name, selector in (("all", mask), ("baseline_correct", correct), ("baseline_bad", bad)):
            self.net[name][0] += float(delta[selector].sum())
            self.net[name][1] += float(selector.sum())

    def add_batch(self, prediction: np.ndarray, baseline: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray, condition: str, scenes: list[str] | None = None) -> None:
        """Aggregate a batch in one NumPy pass; per-sample calls made eval path unacceptably slow."""
        if scenes is None:
            self.statistics.append({"condition": condition, "scene": "batch", "statistics": local_protocol.accuracy_sufficient_statistics(prediction, ground_truth, valid)})
        else:
            if len(scenes) != len(prediction):
                raise ValueError("scene count must match evaluation batch")
            self.statistics.extend(
                {"condition": condition, "scene": scene, "statistics": local_protocol.accuracy_sufficient_statistics(prediction[index], ground_truth[index], valid[index])}
                for index, scene in enumerate(scenes)
            )
        error = np.linalg.norm(prediction - ground_truth, axis=-1)
        base_error = np.linalg.norm(baseline - ground_truth, axis=-1)
        motion = np.linalg.norm(ground_truth, axis=-1)
        mask = valid & np.isfinite(error) & np.isfinite(base_error)
        for name, selector in (("<0.25", base_error < 0.25), ("0.25-0.5", (base_error >= 0.25) & (base_error < 0.5)), ("0.5-1", (base_error >= 0.5) & (base_error < 1)), ("1-3", (base_error >= 1) & (base_error < 3)), ("3-10", (base_error >= 3) & (base_error < 10)), (">10", base_error >= 10)):
            selected = mask & selector
            self.baseline_bins[name][0] += float(error[selected].sum())
            self.baseline_bins[name][1] += float(selected.sum())
        for name, selector in (("s0-10", motion < 10), ("s10-40", (motion >= 10) & (motion < 40)), ("s40+", motion >= 40)):
            selected = mask & selector
            self.motion[name][0] += float(error[selected].sum())
            self.motion[name][1] += float(selected.sum())
        if condition == "clean":
            return
        bad = mask & (base_error > 1)
        correct = mask & (base_error < 0.5)
        self.rescue[0] += float(((base_error - error >= 0.5) & bad).sum())
        self.rescue[1] += float(bad.sum())
        self.harm[0] += float(((error - base_error > 0.25) & correct).sum())
        self.harm[1] += float(correct.sum())
        delta = base_error - error
        for name, selector in (("all", mask), ("baseline_correct", correct), ("baseline_bad", bad)):
            self.net[name][0] += float(delta[selector].sum())
            self.net[name][1] += float(selector.sum())

    def summary(self) -> dict:
        aggregate = local_protocol.summarize_statistics(local_protocol.merge_statistics([row["statistics"] for row in self.statistics]))
        per_condition = local_protocol.group_statistics(self.statistics, "condition")
        per_scene = local_protocol.group_statistics(self.statistics, "scene")
        corrupt_epe = statistics.fmean(value["epe"] for name, value in per_condition.items() if name != "clean")
        motion = {name: (values[0] / values[1] if values[1] else math.nan) for name, values in self.motion.items()}
        return {
            **aggregate,
            "clean_epe": per_condition["clean"]["epe"],
            "corrupt_epe": corrupt_epe,
            "per_condition": per_condition,
            "per_scene": per_scene,
            "per_scene_macro_epe": statistics.fmean(value["epe"] for value in per_scene.values()),
            "motion_epe": motion,
            "rescue_rate": self.rescue[0] / self.rescue[1] if self.rescue[1] else math.nan,
            "harm_rate": self.harm[0] / self.harm[1] if self.harm[1] else math.nan,
            "net_correction": {name: values[0] / values[1] if values[1] else math.nan for name, values in self.net.items()},
            "epe_by_baseline_error": {name: values[0] / values[1] if values[1] else math.nan for name, values in self.baseline_bins.items()},
        }


def _paired_hierarchical_bootstrap(base_records: list[dict], final_records: list[dict], seed: int, repetitions: int = 500) -> dict:
    if len(base_records) != len(final_records) or not base_records:
        raise ValueError("paired bootstrap requires aligned non-empty records")
    scenes: dict[str, list[int]] = defaultdict(list)
    for index, (base, final) in enumerate(zip(base_records, final_records, strict=True)):
        if base["scene"] != final["scene"] or base["condition"] != final["condition"]:
            raise ValueError("paired bootstrap record mismatch")
        scenes[base["scene"]].append(index)
    keys = sorted(scenes)
    rng = np.random.default_rng(seed)
    deltas = []
    percentages = []
    for _ in range(repetitions):
        selected = []
        for scene_index in rng.integers(0, len(keys), size=len(keys)):
            indices = scenes[keys[int(scene_index)]]
            selected.extend(indices[int(index)] for index in rng.integers(0, len(indices), size=len(indices)))
        base_count = sum(int(base_records[index]["statistics"]["valid_count"]) for index in selected)
        final_count = sum(int(final_records[index]["statistics"]["valid_count"]) for index in selected)
        base_epe = sum(float(base_records[index]["statistics"]["epe_sum"]) for index in selected) / base_count
        final_epe = sum(float(final_records[index]["statistics"]["epe_sum"]) for index in selected) / final_count
        deltas.append(final_epe - base_epe)
        percentages.append(100.0 * (final_epe / base_epe - 1.0))
    return {
        "repetitions": repetitions,
        "delta_epe_ci95": np.percentile(deltas, [2.5, 50, 97.5]).tolist(),
        "delta_pct_ci95": np.percentile(percentages, [2.5, 50, 97.5]).tolist(),
    }


def _conv_flop_counter(module: nn.Module):
    total = [0]

    def count(layer: nn.Conv2d, inputs, output):
        batch, out_channels, height, width = output.shape
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels // layer.groups
        total[0] += int(2 * batch * out_channels * height * width * kernel_ops)

    handles = [layer.register_forward_hook(count) for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    return total, handles


def _native_batch(images: list[np.ndarray], device: torch.device, height: int, width: int) -> torch.Tensor:
    rgb = np.stack([image[:, :, ::-1].copy() for image in images])
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().to(device, non_blocking=True)
    return F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=True)


def _full_flow(flow: torch.Tensor, full_height: int, full_width: int, native_height: int, native_width: int) -> np.ndarray:
    result = F.interpolate(flow.float(), size=(full_height, full_width), mode="bilinear", align_corners=True)
    result[:, 0] *= full_width / native_width
    result[:, 1] *= full_height / native_height
    return result.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)


def _full_delta(delta: torch.Tensor, full_height: int, full_width: int, native_height: int, native_width: int) -> np.ndarray:
    return _full_flow(delta, full_height, full_width, native_height, native_width)


def _eval_rows(count: int) -> list[dict]:
    rows = _split_rows(DEV_SCENES)
    selected = FAST.select_balanced(rows, count)
    if len(selected) != count:
        raise RuntimeError(f"evaluation requested {count} rows but selected {len(selected)}")
    return selected


def _confirm_rows(count: int) -> list[dict]:
    selected = FAST.select_balanced(_split_rows(CONFIRM_SCENES), count)
    if len(selected) != count:
        raise RuntimeError(f"confirm requested {count} rows but selected {len(selected)}")
    return selected


def _condition_images(row: dict, condition: str) -> tuple[np.ndarray, np.ndarray]:
    key = (str(row["sample_id"]), condition)
    cached = _CONDITION_IMAGE_CACHE.get(key)
    if cached is not None:
        return cached
    image1 = _read_bgr(row["image1"])
    image2 = _read_bgr(row["image2"])
    if condition == "clean":
        result = (image1, image2)
    else:
        result = FAST.corrupt_pair(image1, image2, condition, FAST.stable_seed(row["sample_id"], condition))
    _CONDITION_IMAGE_CACHE[key] = result
    return result


def _evaluate_head(model: nn.Module | None, head: nn.Module, variant: str, rows: list[dict], device: torch.device, amp_dtype: torch.dtype | None, native_height: int, native_width: int, batch_size: int, output_dir: Path, step: int, precision: str, forward_cache=None) -> dict:
    head.eval()
    progress_path = output_dir.parent / "eval_progress.json"
    base_metrics = MetricAccumulator()
    final_metrics = MetricAccumulator()
    magnitudes = Reservoir(200000, 17)
    gate_pairs = Reservoir(300000, 19)
    gate_errors = Reservoir(300000, 19)
    gate_counts = np.zeros(5, dtype=np.int64)
    gate_error_sums = np.zeros(5, dtype=np.float64)
    gate_behavior = defaultdict(lambda: {"gate_sum": 0.0, "correction_sum": 0.0, "count": 0, "gt_025": 0, "gt_05": 0, "gt_075": 0})
    gate_error_bins = defaultdict(lambda: {"gate_sum": 0.0, "correction_sum": 0.0, "delta_epe_sum": 0.0, "count": 0})
    correction_bins = defaultdict(lambda: [0.0, 0])
    ode_steps = defaultdict(lambda: {"epe": 0.0, "correction_gain": 0.0, "residual_norm": 0.0, "count": 0})
    runtimes = []
    flop_total, flop_handles = _conv_flop_counter(head)
    head_flops = None
    full_height = full_width = None
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            ground_truth = [_read_eval_gt(row["gt"]) for row in batch]
            full_height, full_width = ground_truth[0][0].shape[:2]
            ground_truth_array = np.stack([item[0] for item in ground_truth])
            valid_array = np.stack([item[1] for item in ground_truth])
            scenes = [str(row["scene"]) for row in batch]
            for condition in ALL_CONDITIONS:
                _atomic_json(progress_path, {"variant": variant, "step": step, "batch_start": start, "condition": condition, "completed_conditions": start // batch_size * len(ALL_CONDITIONS) + ALL_CONDITIONS.index(condition) + 1, "total_conditions": ((len(rows) + batch_size - 1) // batch_size) * len(ALL_CONDITIONS)})
                t0 = time.perf_counter()
                if forward_cache is None:
                    images = [_condition_images(row, condition) for row in batch]
                    image1 = _native_batch([pair[0] for pair in images], device, native_height, native_width)
                    image2 = _native_batch([pair[1] for pair in images], device, native_height, native_width)
                    bundle = _forward_bundle(model, image1, image2, device, amp_dtype)
                else:
                    images = image1 = image2 = None
                    bundle = forward_cache.load(condition, start, device)
                output = head(bundle)
                if head_flops is None:
                    head_flops = flop_total[0] / len(batch)
                    for handle in flop_handles:
                        handle.remove()
                prediction = bundle["f0"] + output["delta"]
                baseline_full = _full_flow(bundle["f0"], full_height, full_width, native_height, native_width)
                prediction_full = _full_flow(prediction, full_height, full_width, native_height, native_width)
                delta_full = _full_delta(output["delta"], full_height, full_width, native_height, native_width)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtimes.extend([(time.perf_counter() - t0) * 1000 / len(batch)] * len(batch))
                if "states" in output:
                    for index, state in enumerate(output["states"]):
                        state_full = _full_flow(state, full_height, full_width, native_height, native_width)
                        errors = []
                        gains = []
                        norms = []
                        for j, (_, valid) in enumerate(ground_truth):
                            gt = ground_truth[j][0]
                            mask = valid & np.isfinite(gt).all(axis=-1)
                            e_base = np.linalg.norm(baseline_full[j] - gt, axis=-1)
                            e_state = np.linalg.norm(baseline_full[j] + state_full[j] - gt, axis=-1)
                            errors.append(float(e_state[mask].mean()))
                            gains.append(float((e_base[mask] - e_state[mask]).mean()))
                            norms.append(float(np.linalg.norm(state_full[j], axis=-1)[mask].mean()))
                        values = ode_steps[index + 1]
                        values["epe"] += statistics.fmean(errors)
                        values["correction_gain"] += statistics.fmean(gains)
                        values["residual_norm"] += statistics.fmean(norms)
                        values["count"] += 1
                base_metrics.add_batch(baseline_full, baseline_full, ground_truth_array, valid_array, condition, scenes)
                final_metrics.add_batch(prediction_full, baseline_full, ground_truth_array, valid_array, condition, scenes)
                for j, row in enumerate(batch):
                    gt, valid = ground_truth[j]
                    if "gate" in output:
                        error = np.linalg.norm(prediction_full[j] - gt, axis=-1)
                        base_error = np.linalg.norm(baseline_full[j] - gt, axis=-1)
                        mask = valid & np.isfinite(error) & np.isfinite(base_error)
                        delta_magnitude = np.linalg.norm(delta_full[j], axis=-1)
                        gate = output["gate"][j, 0].float().cpu().numpy()
                        if gate.shape != mask.shape:
                            gate = cv.resize(gate, (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_LINEAR)
                        scope = "clean" if condition == "clean" else "corrupt"
                        behavior = gate_behavior[scope]
                        behavior["gate_sum"] += float(gate[mask].sum())
                        behavior["correction_sum"] += float(delta_magnitude[mask].sum())
                        behavior["count"] += int(mask.sum())
                        behavior["gt_025"] += int((gate[mask] > 0.25).sum())
                        behavior["gt_05"] += int((gate[mask] > 0.5).sum())
                        behavior["gt_075"] += int((gate[mask] > 0.75).sum())
                        for name, selector in (("<0.25", base_error < 0.25), ("0.25-0.5", (base_error >= 0.25) & (base_error < 0.5)), ("0.5-1", (base_error >= 0.5) & (base_error < 1)), ("1-3", (base_error >= 1) & (base_error < 3)), (">3", base_error >= 3)):
                            selected = mask & selector
                            values = gate_error_bins[f"{scope}:{name}"]
                            values["gate_sum"] += float(gate[selected].sum())
                            values["correction_sum"] += float(delta_magnitude[selected].sum())
                            values["delta_epe_sum"] += float((error[selected] - base_error[selected]).sum())
                            values["count"] += int(selected.sum())
                    if condition != "clean":
                        error = np.linalg.norm(prediction_full[j] - gt, axis=-1)
                        base_error = np.linalg.norm(baseline_full[j] - gt, axis=-1)
                        mask = valid & np.isfinite(error) & np.isfinite(base_error)
                        delta_magnitude = np.linalg.norm(delta_full[j], axis=-1)
                        magnitudes.update(delta_magnitude[mask][::16])
                        for name, selector in (("<0.25", base_error < 0.25), ("0.25-0.5", (base_error >= 0.25) & (base_error < 0.5)), ("0.5-1", (base_error >= 0.5) & (base_error < 1)), ("1-3", (base_error >= 1) & (base_error < 3)), ("3-10", (base_error >= 3) & (base_error < 10)), (">10", base_error >= 10)):
                            selected = mask & selector
                            correction_bins[name][0] += float(delta_magnitude[selected].sum())
                            correction_bins[name][1] += int(selected.sum())
                        if "gate" in output:
                            gate = output["gate"][j, 0].float().cpu().numpy()
                            if gate.shape != mask.shape:
                                gate = cv.resize(gate, (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_LINEAR)
                            sampled = mask[::16, ::16]
                            values = gate[::16, ::16][sampled]
                            errors = base_error[::16, ::16][sampled]
                            gate_pairs.update(values)
                            gate_errors.update(errors)
                            bins = np.minimum((values * 5).astype(int), 4)
                            for bin_index in range(5):
                                selected = bins == bin_index
                                gate_counts[bin_index] += int(selected.sum())
                                gate_error_sums[bin_index] += float(errors[selected].sum())
                del images, image1, image2, bundle, output, prediction, baseline_full, prediction_full, delta_full
            del ground_truth, ground_truth_array, valid_array
            if device.type == "cuda":
                torch.cuda.empty_cache()
    final = final_metrics.summary()
    baseline = base_metrics.summary()
    final["proxy_rbs_vs_baseline"] = 0.5 * (final["clean_epe"] / baseline["clean_epe"] + final["corrupt_epe"] / baseline["corrupt_epe"])
    final["delta_clean_pct"] = 100.0 * (final["clean_epe"] / baseline["clean_epe"] - 1.0)
    final["delta_corrupt_pct"] = 100.0 * (final["corrupt_epe"] / baseline["corrupt_epe"] - 1.0)
    final["paired_hierarchical_bootstrap"] = _paired_hierarchical_bootstrap(base_metrics.statistics, final_metrics.statistics, seed=step + 1009)
    final["correction_magnitude_mean"] = float(np.mean(magnitudes.values)) if magnitudes.values else 0.0
    final["correction_magnitude_p95"] = magnitudes.percentile(95)
    final["correction_magnitude_by_baseline_error"] = {name: values[0] / values[1] if values[1] else math.nan for name, values in correction_bins.items()}
    if gate_pairs.values:
        x = np.asarray(gate_pairs.values)
        y = np.asarray(gate_errors.values)
        final["gate_error_spearman"] = _spearman(x, y)
        final["gate_roc_auc_error_gt1"] = _roc_auc(y > 1.0, x)
        final["gate_calibration"] = {
            f"{index / 5:.1f}-{(index + 1) / 5:.1f}": {
                "count": int(gate_counts[index]),
                "mean_baseline_epe": float(gate_error_sums[index] / gate_counts[index]) if gate_counts[index] else math.nan,
            }
            for index in range(5)
        }
        final["gate_analysis"] = {
            "summary": {
                scope: {
                    "mean_gate": values["gate_sum"] / values["count"] if values["count"] else math.nan,
                    "mean_correction": values["correction_sum"] / values["count"] if values["count"] else math.nan,
                    "gate_gt_0.25_coverage": values["gt_025"] / values["count"] if values["count"] else math.nan,
                    "gate_gt_0.5_coverage": values["gt_05"] / values["count"] if values["count"] else math.nan,
                    "gate_gt_0.75_coverage": values["gt_075"] / values["count"] if values["count"] else math.nan,
                }
                for scope, values in sorted(gate_behavior.items())
            },
            "by_baseline_error": {
                scope: {
                    name: {
                        "mean_gate": values["gate_sum"] / values["count"] if values["count"] else math.nan,
                        "mean_correction": values["correction_sum"] / values["count"] if values["count"] else math.nan,
                        "final_delta_epe": values["delta_epe_sum"] / values["count"] if values["count"] else math.nan,
                    }
                    for key, values in sorted(gate_error_bins.items())
                    for key_scope, name in [key.split(":", 1)]
                    if key_scope == scope
                }
                for scope in ("clean", "corrupt")
            },
        }
    if ode_steps:
        final["ode_steps"] = [
            {"step": index, **{name: values[name] / values["count"] for name in ("epe", "correction_gain", "residual_norm")}}
            for index, values in sorted(ode_steps.items())
        ]
        final["ode_field_evaluations"] = 6
        final["ode_field_latency_ms_per_evaluation"] = statistics.median(runtimes) / 6
    final.update({
        "contract_id": CONTRACT_ID,
        "variant": variant,
        "step": step,
        "seed": int(output_dir.parent.name.split("seed")[-1]) if "seed" in output_dir.parent.name else 0,
        "eval_sample_count": len(rows),
        "native_size": [native_height, native_width],
        "params_added": sum(parameter.numel() for parameter in head.parameters()),
        "head_conv_flops_per_image": head_flops,
        "latency_median_ms": statistics.median(runtimes),
        "latency_scope": "cache_replay_head_only" if forward_cache is not None else "live_parent_plus_head",
        "images_per_sec": len(rows) * len(ALL_CONDITIONS) / max(time.perf_counter() - started, 1e-9),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else 0.0,
        "baseline": baseline,
        "precision": precision,
        "protocol": "Spring dev grouped held-out + V2 18-corruption synthetic proxy",
    })
    _atomic_json(output_dir / f"eval_step_{step:04d}.json", final)
    return final


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result
    if len(x) < 2:
        return math.nan
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _run_rdiag(model: nn.Module, device: torch.device, amp_dtype: torch.dtype | None, rows: list[dict], native_height: int, native_width: int, batch_size: int) -> dict:
    pixel_stride = 16
    accumulators = {name: Reservoir(200000, index + 1) for index, name in enumerate(RDIAG_BUCKETS)}
    totals = {name: {"count": 0, "sum": 0.0, "thresholds": defaultdict(int)} for name in accumulators}
    spatial = {ratio: {"baseline": 0.0, "approx": 0.0, "full": 0.0, "count": 0} for ratio in (2, 4, 8)}
    progress_path = OUT / "artifacts" / "rdiag_progress.json"
    total_conditions = math.ceil(len(rows) / batch_size) * len(ALL_CONDITIONS)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            for condition in ALL_CONDITIONS:
                images = [_condition_images(row, condition) for row in batch]
                native1 = _native_batch([pair[0] for pair in images], device, native_height, native_width)
                native2 = _native_batch([pair[1] for pair in images], device, native_height, native_width)
                bundle = _forward_bundle(model, native1, native2, device, amp_dtype)
                ground_truth = [_read_eval_gt(row["gt"]) for row in batch]
                full_height, full_width = ground_truth[0][0].shape[:2]
                base = _full_flow(bundle["f0"], full_height, full_width, native_height, native_width)
                gt_stack = torch.from_numpy(np.stack([item[0] for item in ground_truth])).permute(0, 3, 1, 2).float().to(device)
                base_tensor = torch.from_numpy(base).permute(0, 3, 1, 2).float().to(device)
                residual_tensor = gt_stack - base_tensor
                for j, row in enumerate(batch):
                    gt, valid = ground_truth[j]
                    residual = residual_tensor[j].float().cpu().permute(1, 2, 0).numpy()
                    sampled_gt = gt[::pixel_stride, ::pixel_stride]
                    sampled_base = base[j][::pixel_stride, ::pixel_stride]
                    sampled_residual = residual[::pixel_stride, ::pixel_stride]
                    sampled_valid = valid[::pixel_stride, ::pixel_stride]
                    base_error = np.linalg.norm(sampled_base - sampled_gt, axis=-1)
                    residual_norm = np.linalg.norm(sampled_residual, axis=-1)
                    finite = sampled_valid & np.isfinite(residual_norm) & np.isfinite(base_error)
                    names = ["clean"] if condition == "clean" else ["corrupt", GROUP_FOR[condition]]
                    motion = np.linalg.norm(sampled_gt, axis=-1)
                    selectors = {
                        "motion_0-10": motion < 10,
                        "motion_10-40": (motion >= 10) & (motion < 40),
                        "motion_40+": motion >= 40,
                        "error_<0.25": base_error < 0.25,
                        "error_0.25-0.5": (base_error >= 0.25) & (base_error < 0.5),
                        "error_0.5-1": (base_error >= 0.5) & (base_error < 1),
                        "error_1-3": (base_error >= 1) & (base_error < 3),
                        "error_>3": base_error >= 3,
                    }
                    for name, selector in selectors.items():
                        if name in accumulators:
                            names.append(name)
                    for name in names:
                        if not name:
                            continue
                        selected = residual_norm[finite]
                        if name in selectors:
                            selected = residual_norm[finite & selectors[name]]
                        if not len(selected):
                            continue
                        accumulators[name].update(selected)
                        total = totals[name]
                        total["count"] += int(len(selected))
                        total["sum"] += float(selected.sum())
                        for threshold in (0.25, 0.5, 1.0, 3.0):
                            total["thresholds"][f"lt_{threshold:g}"] += int((selected < threshold).sum())
                            total["thresholds"][f"gt_{threshold:g}"] += int((selected > threshold).sum())
                    for ratio in (2, 4, 8):
                        low = F.interpolate(residual_tensor[j:j + 1], scale_factor=1 / ratio, mode="bilinear", align_corners=False) / ratio
                        approx = F.interpolate(low, size=(full_height, full_width), mode="bilinear", align_corners=False) * ratio
                        approx_np = approx[0].cpu().permute(1, 2, 0).numpy()
                        base_epe = np.linalg.norm(sampled_base - sampled_gt, axis=-1)
                        approx_epe = np.linalg.norm((base[j] + approx_np)[::pixel_stride, ::pixel_stride] - sampled_gt, axis=-1)
                        full_epe = np.linalg.norm(sampled_base + sampled_residual - sampled_gt, axis=-1)
                        mask = sampled_valid & np.isfinite(base_epe) & np.isfinite(approx_epe)
                        spatial[ratio]["baseline"] += float(base_epe[mask].sum())
                        spatial[ratio]["approx"] += float(approx_epe[mask].sum())
                        spatial[ratio]["full"] += float(full_epe[mask].sum())
                        spatial[ratio]["count"] += int(mask.sum())
                _atomic_json(progress_path, {"complete": False, "batch_start": start, "condition": condition, "completed_conditions": start // batch_size * len(ALL_CONDITIONS) + ALL_CONDITIONS.index(condition) + 1, "total_conditions": total_conditions})
    def summarize(name: str) -> dict:
        total = totals[name]
        count = total["count"]
        values = accumulators[name]
        return {
            "count": count,
            "mean": total["sum"] / count if count else math.nan,
            "median": values.percentile(50),
            "p90": values.percentile(90),
            "p95": values.percentile(95),
            "p99": values.percentile(99),
            "probabilities": {
                "lt_0.25": total["thresholds"]["lt_0.25"] / count if count else math.nan,
                "lt_0.50": total["thresholds"]["lt_0.5"] / count if count else math.nan,
                "lt_1.00": total["thresholds"]["lt_1"] / count if count else math.nan,
                "gt_1.00": total["thresholds"]["gt_1"] / count if count else math.nan,
                "gt_3.00": total["thresholds"]["gt_3"] / count if count else math.nan,
            },
            "reservoir_count": values.seen,
        }
    result = {
        "contract_id": CONTRACT_ID,
        "complete": True,
        "parent_checkpoint_sha256": _sha256(PARENT_CHECKPOINT),
        "sample_count": len(rows),
        "native_size": [native_height, native_width],
        "diagnostic_pixel_stride": pixel_stride,
        "conditions": list(ALL_CONDITIONS),
        "clean": summarize("clean"),
        "corrupt": summarize("corrupt"),
        "by_corruption_group": {name: summarize(name) for name in GROUPS},
        "by_motion": {name: summarize(name) for name in ("motion_0-10", "motion_10-40", "motion_40+")},
        "by_baseline_error": {name: summarize(name) for name in ("error_<0.25", "error_0.25-0.5", "error_0.5-1", "error_1-3", "error_>3")},
        "spatial_compression": {
            str(ratio): {
                **values,
                "baseline_epe": values["baseline"] / values["count"] if values["count"] else math.nan,
                "approx_epe": values["approx"] / values["count"] if values["count"] else math.nan,
                "full_epe": values["full"] / values["count"] if values["count"] else math.nan,
                "correction_gain_explained": (values["baseline"] - values["approx"]) / max(values["baseline"] - values["full"], 1e-12),
            }
            for ratio, values in spatial.items()
        },
    }
    _atomic_json(progress_path, {"complete": True, "completed_conditions": total_conditions, "total_conditions": total_conditions})
    histogram_dir = OUT / "artifacts" / "residual_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)
    for name, reservoir in accumulators.items():
        if reservoir.values:
            counts, edges = np.histogram(np.asarray(reservoir.values), bins=np.linspace(0, 10, 101))
            _atomic_json(histogram_dir / f"{name.replace('/', '_')}.json", {"seen": reservoir.seen, "edges": edges.tolist(), "counts": counts.tolist()})
    return result


def _assert_default_equivalence(model: nn.Module, device: torch.device, amp_dtype: torch.dtype | None, height: int, width: int) -> dict:
    image1 = torch.rand(1, 3, height, width, device=device) * 255
    image2 = torch.rand(1, 3, height, width, device=device) * 255
    with torch.no_grad(), _amp(device, amp_dtype):
        default = model(image1, image2)
        aux_off = model(image1, image2, return_aux=False)
    difference = float((default["flow"][-1] - aux_off["flow"][-1]).abs().max().float().cpu())
    if difference != 0.0:
        raise AssertionError(f"return_aux=False changed WAFT output: max_abs={difference}")
    return {"max_abs_flow_difference": difference, "passed": True}


def _load_base(device: torch.device, precision: str = "bf16"):
    predictor = adapters.load_predictor(PARENT_MODEL, device, precision)
    model = predictor.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return predictor, model


def _autotune(model: nn.Module, variant: str, device: torch.device, amp_dtype: torch.dtype | None, height: int, width: int, target_gib: tuple[float, float], max_batch: int, seed: int) -> dict:
    if device.type != "cuda":
        return {"batch_size": 1, "tests": [], "stable_30": True, "note": "CPU fallback"}
    low_target, high_target = target_gib
    tests = []

    def test(batch_size: int, steps: int) -> tuple[bool, dict]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(seed + batch_size + steps)
        head = _make_head(variant).to(device).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-5)
        image1 = torch.rand(batch_size, 3, height, width, device=device) * 255
        image2 = torch.rand(batch_size, 3, height, width, device=device) * 255
        ground_truth = torch.rand(batch_size, 2, height, width, device=device)
        valid = torch.ones(batch_size, height, width, dtype=torch.bool, device=device)
        try:
            for _ in range(steps):
                _step(model, head, optimizer, (image1, image2, ground_truth, valid), device, amp_dtype)
            torch.cuda.synchronize(device)
            record = {
                "batch_size": batch_size,
                "steps": steps,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            return record["peak_reserved_gib"] <= high_target, record
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            return False, {"batch_size": batch_size, "steps": steps, "oom": True}
        finally:
            del head, optimizer, image1, image2, ground_truth, valid
            gc.collect()
            torch.cuda.empty_cache()

    safe = 0
    failed = None
    candidate = 1
    while candidate <= max_batch:
        passed, record = test(candidate, 1)
        tests.append(record)
        if not passed:
            failed = candidate
            break
        safe = candidate
        if record["peak_reserved_gib"] >= low_target:
            break
        candidate *= 2
    if failed and safe and failed - safe > 1:
        left, right = safe + 1, failed - 1
        while left <= right:
            candidate = (left + right) // 2
            passed, record = test(candidate, 1)
            tests.append(record)
            if passed:
                safe = candidate
                left = candidate + 1
            else:
                right = candidate - 1
    if safe == 0:
        safe = 1
    passed, stable = test(safe, 30)
    tests.append(stable)
    if not passed:
        for fallback in range(safe - 1, 0, -1):
            passed, stable = test(fallback, 30)
            tests.append(stable)
            if passed:
                safe = fallback
                break
    return {
        "batch_size": safe,
        "head_precision": "fp32",
        "target_reserved_gib": [low_target, high_target],
        "tests": tests,
        "stable_30": passed,
        "independent_per_variant": True,
    }


class AuxForward(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image1: torch.Tensor, image2: torch.Tensor):
        output = self.model(image1, image2, return_aux=True)
        aux = output["aux"]
        return output["flow"][-1], aux["fmap1"], aux["fmap2"], aux["hidden"], aux["flow_low"], aux["last_update"]


def _compile_benchmark(model: nn.Module, device: torch.device, amp_dtype: torch.dtype | None, height: int, width: int) -> dict:
    result = {"attempted": False, "use_compile": False}
    if device.type != "cuda" or not hasattr(torch, "compile"):
        result["reason"] = "CUDA or torch.compile unavailable"
        return result
    result["attempted"] = True
    image1 = torch.rand(1, 3, height, width, device=device) * 255
    image2 = torch.rand(1, 3, height, width, device=device) * 255

    def run(forward, count: int) -> float:
        values = []
        with torch.no_grad():
            for index in range(count):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with _amp(device, amp_dtype):
                    forward(image1, image2)
                torch.cuda.synchronize(device)
                if index >= 5:
                    values.append(time.perf_counter() - start)
        return statistics.median(values)

    try:
        eager = run(model, 50)
        compiled_model = torch.compile(AuxForward(model), dynamic=False)
        compiled = run(compiled_model, 50)
        result.update({"eager_median_s": eager, "compile_median_s": compiled, "improvement_pct": 100 * (eager / compiled - 1), "use_compile": compiled <= eager * 0.95})
        del compiled_model
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        del image1, image2
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _train_variant(args, variant: str, seed: int, parent_dir: Path, use_compile: bool = False, forward_cache=None) -> dict:
    device = torch.device(args.device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_predictor, model = _load_base(device, args.precision)
    amp_dtype = base_predictor.amp_dtype
    if use_compile:
        model = torch.compile(AuxForward(model), dynamic=False)
        model._residual_tuple_output = True
    head = _make_head(variant).to(device).train()
    run_dir = parent_dir / f"{variant.lower()}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    autotune_path = run_dir / "autotune.json"
    try:
        autotune_record = json.loads(autotune_path.read_text())
    except (OSError, json.JSONDecodeError):
        autotune_record = {}
    if args.batch_size > 0:
        autotune = {"batch_size": args.batch_size, "stable_30": True, "head_precision": "fp32", "mode": "fixed_completion_first"}
        _atomic_json(autotune_path, autotune)
    elif autotune_record.get("stable_30") and autotune_record.get("head_precision") == "fp32":
        autotune = autotune_record
    else:
        autotune = _autotune(model, variant, device, amp_dtype, args.native_height, args.native_width, (72.0, 76.0), args.max_batch_size, seed)
        _atomic_json(autotune_path, autotune)
    if not autotune.get("stable_30"):
        raise RuntimeError(f"autotune found no stable batch for {variant}: {autotune}")
    batch_size = int(autotune["batch_size"])
    rows = _split_rows(TRAIN_SCENES)
    dataset = SpringResidualDataset(rows, args.steps * batch_size, args.native_height, args.native_width, seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state = {"step": 0, "step_time_s": 0.0, "data_time_s": 0.0, "forward_time_s": 0.0, "backward_time_s": 0.0}
    previous_status = {}
    try:
        previous_status = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    resume_payload = None
    resume_step = 0
    if previous_status.get("complete") is not True:
        for checkpoint_path in sorted(run_dir.glob("step_*.pt"), reverse=True):
            try:
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                checkpoint_step = int(payload.get("steps", -1))
                if (
                    payload.get("variant") == variant
                    and int(payload.get("seed", -1)) == seed
                    and payload.get("contract_id") == CONTRACT_ID
                    and checkpoint_step > 0
                    and checkpoint_step <= args.steps
                    and int(payload.get("total_steps", -1)) == args.steps
                    and int(payload.get("batch_size", -1)) == batch_size
                    and payload.get("precision") == args.precision
                    and payload.get("parent_checkpoint_sha256") == _sha256(PARENT_CHECKPOINT)
                ):
                    resume_payload = payload
                    resume_step = checkpoint_step
                    break
            except (OSError, ValueError, RuntimeError, TypeError):
                continue
    dataset.skip(resume_step * batch_size)
    resumed_nonfinite = int(previous_status.get("nonfinite_steps", 0)) if resume_step else 0
    _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": resume_step, "complete": False, "batch_size": batch_size, "nonfinite_steps": resumed_nonfinite, "resumed_from": resume_step})
    watchdog = Watchdog(OUT / "gpu_profile.csv", variant + f"_seed{seed}", state)
    watchdog.start()
    eval_rows = _eval_rows(args.eval_samples)
    eval_dir = run_dir / "evaluations"
    initial_path = eval_dir / "eval_step_0000.json"
    try:
        initial_report = json.loads(initial_path.read_text())
        initial_valid = (
            initial_report.get("contract_id") == CONTRACT_ID
            and int(initial_report.get("eval_sample_count", -1)) == len(eval_rows)
            and initial_report.get("native_size") == [args.native_height, args.native_width]
            and initial_report.get("precision") == args.precision
        )
    except (OSError, ValueError, json.JSONDecodeError):
        initial_valid = False
    if not initial_valid:
        initial = _evaluate_head(model, head, variant, eval_rows, device, amp_dtype, args.native_height, args.native_width, args.eval_batch_size, eval_dir, 0, args.precision, forward_cache)
        initial["seed"] = seed
        _atomic_torch(run_dir / "step_0000.pt", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "parent_checkpoint_sha256": _sha256(PARENT_CHECKPOINT), "state_dict": head.state_dict(), "optimizer": optimizer.state_dict(), "steps": 0, "total_steps": args.steps, "batch_size": batch_size, "precision": args.precision})
    if resume_payload is not None:
        head.load_state_dict(resume_payload["state_dict"])
        if resume_payload.get("optimizer"):
            optimizer.load_state_dict(resume_payload["optimizer"])
    head.train()
    losses = [float(previous_status["last_loss"])] if resume_step and "last_loss" in previous_status else []
    nonfinite_steps = resumed_nonfinite
    iterator = iter(loader)
    state["step"] = resume_step
    last = time.perf_counter()
    try:
        for step in range(resume_step + 1, args.steps + 1):
            state["step"] = step
            data_started = time.perf_counter()
            batch = next(iterator)
            data_time = time.perf_counter() - data_started
            forward_started = time.perf_counter()
            image1, image2, ground_truth, valid = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            try:
                bundle = _forward_bundle(model, image1, image2, device, amp_dtype)
                output = head(bundle)
                loss, _ = _loss(head, bundle, ground_truth, valid, output)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite residual loss: {float(loss.detach().cpu())}")
                forward_time = time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
                    raise FloatingPointError("non-finite residual gradient")
            except FloatingPointError as error:
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": step, "complete": False, "batch_size": batch_size, "nonfinite_steps": nonfinite_steps, "last_error": str(error)})
                print(f"[{variant}/seed{seed}] skip step={step} ({error}); nonfinite_steps={nonfinite_steps}", flush=True)
                if nonfinite_steps > args.max_nonfinite_steps:
                    raise
                continue
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            backward_time = time.perf_counter() - backward_started
            losses.append(float(loss.detach().float().cpu()))
            step_time = time.perf_counter() - last
            last = time.perf_counter()
            state.update({"step": step, "step_time_s": step_time, "data_time_s": data_time, "forward_time_s": forward_time, "backward_time_s": backward_time})
            if step == 1 or step % args.log_every == 0:
                print(f"[{variant}/seed{seed}] step={step}/{args.steps} loss={losses[-1]:.6f} batch={batch_size}", flush=True)
                _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": step, "complete": False, "last_loss": losses[-1], "batch_size": batch_size, "nonfinite_steps": nonfinite_steps})
            if step % args.eval_every == 0 or step == args.steps:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                report = _evaluate_head(model, head, variant, eval_rows, device, amp_dtype, args.native_height, args.native_width, args.eval_batch_size, eval_dir, step, args.precision, forward_cache)
                report["seed"] = seed
                _atomic_json(run_dir / "latest_eval.json", report)
                _atomic_torch(run_dir / f"step_{step:04d}.pt", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "parent_checkpoint_sha256": _sha256(PARENT_CHECKPOINT), "state_dict": head.state_dict(), "optimizer": optimizer.state_dict(), "steps": step, "total_steps": args.steps, "batch_size": batch_size, "precision": args.precision})
                _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": step, "complete": step == args.steps, "last_loss": losses[-1], "batch_size": batch_size, "nonfinite_steps": nonfinite_steps})
        terminal_status = json.loads((run_dir / "status.json").read_text())
        if terminal_status.get("complete") is not True and state["step"] == args.steps and nonfinite_steps <= args.max_nonfinite_steps:
            report = _evaluate_head(model, head, variant, eval_rows, device, amp_dtype, args.native_height, args.native_width, args.eval_batch_size, eval_dir, args.steps, args.precision, forward_cache)
            report["seed"] = seed
            _atomic_json(run_dir / "latest_eval.json", report)
            _atomic_torch(run_dir / f"step_{args.steps:04d}.pt", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "parent_checkpoint_sha256": _sha256(PARENT_CHECKPOINT), "state_dict": head.state_dict(), "optimizer": optimizer.state_dict(), "steps": args.steps, "total_steps": args.steps, "batch_size": batch_size, "precision": args.precision})
            _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": args.steps, "complete": True, "last_loss": losses[-1] if losses else None, "batch_size": batch_size, "nonfinite_steps": nonfinite_steps})
    except Exception as error:
        _atomic_json(run_dir / "status.json", {"contract_id": CONTRACT_ID, "variant": variant, "seed": seed, "step": state["step"], "complete": False, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        watchdog.close()
        del loader, iterator, optimizer, head, model, base_predictor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return json.loads((run_dir / "latest_eval.json").read_text())


def _write_registry_csv(registry: dict, reports: dict[str, dict]) -> None:
    path = OUT / "experiment_registry.csv"
    fields = ["run_id", "variant", "seed", "status", "clean_epe", "corrupt_epe", "proxy_rbs_vs_baseline", "delta_clean_pct", "delta_corrupt_pct", "params_added", "latency_median_ms", "images_per_sec", "peak_vram_mb"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_id, report in reports.items():
            writer.writerow({"run_id": run_id, "variant": report.get("variant"), "seed": report.get("seed", 0), "status": "complete", **{field: report.get(field) for field in fields[4:]}})


def _reuse_complete_run(run_dir: Path, steps: int, batch_size: int, eval_samples: int, native_size: list[int], precision: str) -> dict | None:
    status_path = run_dir / "status.json"
    report_path = run_dir / "latest_eval.json"
    checkpoint_path = run_dir / f"step_{steps:04d}.pt"
    if not (status_path.exists() and report_path.exists() and checkpoint_path.exists()):
        return None
    try:
        status = json.loads(status_path.read_text())
        report = json.loads(report_path.read_text())
        if status.get("complete") is not True or status.get("contract_id") != CONTRACT_ID or int(status.get("step", -1)) != steps or (batch_size > 0 and int(status.get("batch_size", -1)) != batch_size):
            return None
        if report.get("contract_id") != CONTRACT_ID or report.get("step") != steps or report.get("variant") != status.get("variant") or int(report.get("seed", -1)) != int(status.get("seed", -2)) or int(report.get("eval_sample_count", -1)) != eval_samples or report.get("native_size") != native_size or report.get("precision") != precision:
            return None
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("contract_id") != CONTRACT_ID
            or payload.get("variant") != status.get("variant")
            or int(payload.get("seed", -1)) != int(status.get("seed", -2))
            or int(payload.get("steps", -1)) != steps
            or int(payload.get("total_steps", -1)) != steps
            or (batch_size > 0 and int(payload.get("batch_size", -1)) != batch_size)
            or payload.get("precision") != precision
            or payload.get("parent_checkpoint_sha256") != _sha256(PARENT_CHECKPOINT)
        ):
            return None
        return report
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError, TypeError):
        return None


class _ZeroHead(nn.Module):
    def forward(self, bundle):
        return {"delta": torch.zeros_like(bundle["f0"])}


def _screen_grade(report: dict) -> str:
    baseline = report["baseline"]["per_condition"]
    final = report["per_condition"]
    improved = sum(
        statistics.fmean(final[name]["epe"] for name in conditions) < statistics.fmean(baseline[name]["epe"] for name in conditions)
        for conditions in GROUPS.values()
    )
    non_degraded = sum(
        statistics.fmean(final[name]["epe"] for name in conditions) <= statistics.fmean(baseline[name]["epe"] for name in conditions)
        for conditions in GROUPS.values()
    )
    if report["delta_clean_pct"] > 0.5 or report["delta_corrupt_pct"] > -0.5 or improved < 2:
        return "REJECT"
    if report["delta_corrupt_pct"] <= -2.0 and non_degraded >= 4:
        return "STRONG_ADVANCE"
    return "WEAK_ADVANCE"


def _evaluate_confirm(args, top2: list[str], cache_module) -> dict:
    try:
        existing = json.loads((OUT / "confirm_summary.json").read_text())
        existing_cache = json.loads((OUT / "cache" / "confirm" / "manifest.json").read_text())
        if existing.get("contract_id") == CONTRACT_ID and existing.get("complete") is True and existing.get("sample_count") == 200 and existing.get("selection", {}).get("top2") == top2 and len(existing.get("candidates", {})) == 2 * len(top2) and existing_cache.get("complete") is True and existing_cache.get("parity", {}).get("passed") is True:
            return existing
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    selection = {"contract_id": CONTRACT_ID, "top2": top2, "source": "dev64 seed0 ranking, then seed1 replication", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _atomic_json(OUT / "confirm_selection.json", selection)
    rows = _confirm_rows(200)
    cache = cache_module._build_forward_cache(args, rows, OUT / "cache" / "confirm", "confirm-forward")
    device = torch.device(args.device)
    baseline = _evaluate_head(None, _ZeroHead().to(device), "B0", rows, device, None, args.native_height, args.native_width, args.eval_batch_size, OUT / "confirm" / "baseline", args.steps, args.precision, cache)
    candidates = {}
    for variant in top2:
        for seed, parent in ((0, OUT / "runs"), (1, OUT / "top2_seed1")):
            checkpoint_path = parent / f"{variant.lower()}_seed{seed}" / f"step_{args.steps:04d}.pt"
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if payload.get("contract_id") != CONTRACT_ID or payload.get("variant") != variant or int(payload.get("seed", -1)) != seed or int(payload.get("steps", -1)) != args.steps:
                raise RuntimeError(f"confirm checkpoint contract mismatch: {checkpoint_path}")
            head = _make_head(variant).to(device).eval()
            head.load_state_dict(payload["state_dict"])
            report = _evaluate_head(None, head, variant, rows, device, None, args.native_height, args.native_width, args.eval_batch_size, OUT / "confirm" / f"{variant.lower()}_seed{seed}", args.steps, args.precision, cache)
            candidates[f"{variant.lower()}_seed{seed}"] = report
            del head, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
    passes = {
        variant: all(
            candidates[f"{variant.lower()}_seed{seed}"]["delta_corrupt_pct"] <= -0.5
            and candidates[f"{variant.lower()}_seed{seed}"]["delta_clean_pct"] <= 0.5
            for seed in (0, 1)
        )
        for variant in top2
    }
    result = {"contract_id": CONTRACT_ID, "complete": True, "selection": selection, "sample_count": len(rows), "baseline": baseline, "candidates": candidates, "both_seed_confirm_pass": passes}
    _atomic_json(OUT / "confirm_summary.json", result)
    return result


def _write_final_comparison(reports: dict[str, dict], registry: dict, rdiag: dict, compile_result: dict, top2: list[str], confirm: dict) -> None:
    rows = []
    for run_id, report in reports.items():
        baseline = report["baseline"]
        rows.append((run_id, report, baseline))
    rows.sort(key=lambda item: (item[1].get("proxy_rbs_vs_baseline", math.inf), item[0]))
    lines = [
        "# WAFT-DAv2-A2 residual screen",
        "",
        "All labeled EPE numbers below use the fixed Stage2.5 dev grouped split and the V2 18-corruption synthetic proxy. Synthetic proxy is not exact RobustSpring scoring.",
        "",
        f"Parent checkpoint SHA256: `{registry['parent']['checkpoint_sha256']}`",
        f"R-DIAG: `artifacts/residual_diag.json`; compile decision: `{compile_result}`",
        "",
        "| Model | Clean EPE | Δ Clean | Corrupt EPE | Δ Corrupt | Proxy RbS | 1px | s0-10 | s10-40 | s40+ | Color | Blur | Noise | Quality | Weather | Rescue | Harm | Params | VRAM MB | images/s | latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id, report, baseline in rows:
        groups = report["per_condition"]
        lines.append(
            "| {run} | {clean:.6f} | {dc:+.3f}% | {corr:.6f} | {dr:+.3f}% | {rbs:.6f} | {one:.3f} | {s0:.6f} | {s1:.6f} | {s4:.6f} | {color:.6f} | {blur:.6f} | {noise:.6f} | {quality:.6f} | {weather:.6f} | {rescue:.4f} | {harm:.4f} | {params} | {vram:.1f} | {speed:.3f} | {lat:.2f} |".format(
                run=run_id,
                clean=report["clean_epe"],
                dc=report["delta_clean_pct"],
                corr=report["corrupt_epe"],
                dr=report["delta_corrupt_pct"],
                rbs=report["proxy_rbs_vs_baseline"],
                one=report["one_px"],
                s0=report["motion_epe"].get("s0-10", math.nan),
                s1=report["motion_epe"].get("s10-40", math.nan),
                s4=report["motion_epe"].get("s40+", math.nan),
                color=statistics.fmean(groups[name]["epe"] for name in GROUPS["color"]),
                blur=statistics.fmean(groups[name]["epe"] for name in GROUPS["blur"]),
                noise=statistics.fmean(groups[name]["epe"] for name in GROUPS["noise"]),
                quality=statistics.fmean(groups[name]["epe"] for name in GROUPS["quality"]),
                weather=statistics.fmean(groups[name]["epe"] for name in GROUPS["weather"]),
                rescue=report["rescue_rate"],
                harm=report["harm_rate"],
                params=report["params_added"],
                vram=report["peak_vram_mb"],
                speed=report["images_per_sec"],
                lat=report["latency_median_ms"],
            )
        )
    grades = {run_id: _screen_grade(report) for run_id, report, _ in rows}
    labels = {}
    for name in VARIANTS:
        seed0 = grades.get(f"{name.lower()}_seed0", "REJECT")
        if name in top2:
            seed1 = grades.get(f"{name.lower()}_seed1", "REJECT")
            if seed0 != "REJECT" and seed1 != "REJECT":
                decision = "ADVANCE"
            elif seed0 != seed1:
                decision = "UNSTABLE"
            else:
                decision = "REJECT"
        else:
            decision = "ADVANCE" if seed0 != "REJECT" else "REJECT"
        labels[name] = decision
    validated = any(labels[name] == "ADVANCE" and confirm["both_seed_confirm_pass"].get(name, False) for name in top2)
    final_decision = "RESIDUAL_HYPOTHESIS_VALIDATED" if validated else "RESIDUAL_HYPOTHESIS_NOT_VALIDATED"
    lines.extend(["", "## Screen decision", ""])
    for name in VARIANTS:
        label = {"R1": "DETERMINISTIC_RESIDUAL", "R2": "RISK_GATING", "R3": "LOW_RES_RESIDUAL", "R4": "RESIDUAL_ODE"}[name]
        lines.append(f"- `{label}`: `{labels[name]}`")
    lines.extend([
        "",
        "- `FLOW_MATCHING`: `SKIPPED_GPU_COUNT_LT5`",
        "",
        f"Confirm200 both-seed pass map: `{confirm['both_seed_confirm_pass']}`.",
        "The real RobustSpring track, if present in per-run artifacts, is stability only because public corrupted GT is unavailable.",
        "",
        "## R-DIAG spatial compression",
        "",
        "```json",
        json.dumps(rdiag["spatial_compression"], indent=2, sort_keys=True),
        "```",
        "",
        final_decision,
    ])
    _atomic_json(OUT / "screen_decision.json", {"decision": final_decision, "grades": grades, "variant_labels": labels, "top2": top2, "confirm_pass": confirm["both_seed_confirm_pass"]})
    (OUT / "final_comparison.md").write_text("\n".join(lines) + "\n")


def run_all(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUT / "run_process.json", {"pid": os.getpid(), "contract_id": CONTRACT_ID, "argv": sys.argv, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    registry = build_registry(args.eval_samples)
    registry["execution"] = {"argv": sys.argv, "arguments": vars(args)}
    smoke = json.loads((OUT / "preflight_smoke.json").read_text())
    if smoke.get("contract_id") != CONTRACT_ID or smoke.get("passed") is not True or smoke.get("batch_size") != args.batch_size or smoke.get("native_size") != [args.native_height, args.native_width] or smoke.get("precision") != args.precision or smoke.get("parent_checkpoint_sha256") != registry["parent"]["checkpoint_sha256"]:
        raise RuntimeError("preflight smoke artifact does not match the production contract")
    registry["checks"] = {"real_path_smoke": smoke}
    _atomic_json(OUT / "experiment_registry.json", registry)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    predictor, model = _load_base(device, args.precision)
    aux_check = _assert_default_equivalence(model, device, predictor.amp_dtype, args.native_height, args.native_width)
    registry["checks"].update({"return_aux_default_equivalence": aux_check, "all_parent_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters())})
    _atomic_json(OUT / "experiment_registry.json", registry)
    _write_block_review("block0_preflight", {
        "contract_v2": registry["protocol"]["contract_id"] == CONTRACT_ID,
        "split_disjoint": not (set(row["scene"] for row in _split_rows(TRAIN_SCENES)) & set(row["scene"] for row in _split_rows(DEV_SCENES) + _split_rows(CONFIRM_SCENES))),
        "parent_frozen": registry["checks"]["all_parent_parameters_frozen"],
        "aux_default_equivalent": registry["checks"]["return_aux_default_equivalence"]["passed"],
        "real_batch4_smoke": smoke["passed"],
    })
    rdiag_path = OUT / "artifacts/residual_diag.json"
    rdiag_histograms = [OUT / "artifacts" / "residual_histograms" / f"{name}.json" for name in RDIAG_BUCKETS]
    try:
        rdiag = json.loads(rdiag_path.read_text())
        rdiag_valid = rdiag.get("complete") is True and rdiag.get("contract_id") == CONTRACT_ID and rdiag.get("sample_count") == args.eval_samples and rdiag.get("native_size") == [args.native_height, args.native_width] and rdiag.get("conditions") == list(ALL_CONDITIONS) and all(path.exists() for path in rdiag_histograms)
    except (OSError, ValueError, json.JSONDecodeError):
        rdiag_valid = False
    if not rdiag_valid:
        rdiag = _run_rdiag(model, device, predictor.amp_dtype, _eval_rows(args.eval_samples), args.native_height, args.native_width, args.eval_batch_size)
        _atomic_json(rdiag_path, rdiag)
    _atomic_json(OUT / "rdiag.json", rdiag)
    del model, predictor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    from experiments.crossdomain_screen import crossdomain_experiment as cache_module
    forward_cache = cache_module._build_forward_cache(args)
    try:
        baseline = json.loads((OUT / "baseline.json").read_text())
        baseline_valid = baseline.get("contract_id") == CONTRACT_ID and baseline.get("eval_sample_count") == args.eval_samples and baseline.get("native_size") == [args.native_height, args.native_width] and baseline.get("precision") == args.precision
    except (OSError, ValueError, json.JSONDecodeError):
        baseline_valid = False
    if not baseline_valid:
        baseline = _evaluate_head(None, _ZeroHead().to(device), "B0", _eval_rows(args.eval_samples), device, None, args.native_height, args.native_width, args.eval_batch_size, OUT / "baseline_eval", 0, args.precision, forward_cache)
        _atomic_json(OUT / "baseline.json", baseline)
    _atomic_json(OUT / "r5_status.json", {"contract_id": CONTRACT_ID, "status": "SKIPPED_GPU_COUNT_LT5", "reason": "R5 requires the source document's five-GPU launch gate; this machine has one A100"})
    cache_manifest = json.loads((cache_module.FORWARD_CACHE_DIR / "manifest.json").read_text())
    _write_block_review("block1_baseline_rdiag_cache", {
        "baseline_dev64": baseline.get("eval_sample_count") == args.eval_samples,
        "rdiag_complete": rdiag.get("complete") is True and rdiag.get("contract_id") == CONTRACT_ID,
        "geometric_corruptions_excluded": not (EXCLUDED_CORRUPTIONS & set(rdiag.get("conditions", []))),
        "forward_cache_complete": cache_manifest.get("complete") is True,
        "forward_cache_parity": cache_manifest.get("parity", {}).get("passed") is True,
    })
    compile_path = OUT / "compile_benchmark.json"
    compile_result = {"attempted": False, "use_compile": False, "reason": "completion-first eager default"}
    if args.compile_benchmark:
        predictor, model = _load_base(device, args.precision)
        compile_result = _compile_benchmark(model, device, predictor.amp_dtype, args.native_height, args.native_width)
        del model, predictor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _atomic_json(compile_path, compile_result)
    order = ("R3", "R2", "R1", "R4") if torch.cuda.device_count() == 1 else VARIANTS
    r1_params = sum(parameter.numel() for parameter in _make_head("R1").parameters())
    reports = {}
    for variant in order:
        run_dir = OUT / "runs" / f"{variant.lower()}_seed0"
        report = _reuse_complete_run(run_dir, args.steps, args.batch_size, args.eval_samples, [args.native_height, args.native_width], args.precision)
        if report is not None:
            print(f"[{variant}/seed0] reuse complete artifact at {run_dir}", flush=True)
        else:
            report = _train_variant(args, variant, 0, OUT / "runs", bool(compile_result.get("use_compile")), forward_cache)
        reports[f"{variant.lower()}_seed0"] = report
        _atomic_json(OUT / f"{variant}_seed0.json", report)
        _write_registry_csv(registry, reports)
        status = json.loads((OUT / "runs" / f"{variant.lower()}_seed0" / "status.json").read_text())
        _write_block_review(f"block2_{variant.lower()}_seed0", {
            "contract": report.get("contract_id") == CONTRACT_ID and status.get("contract_id") == CONTRACT_ID,
            "terminal_step": int(report.get("step", -1)) == args.steps and int(status.get("step", -1)) == args.steps,
            "complete": status.get("complete") is True,
            "batch4": int(status.get("batch_size", -1)) == args.batch_size,
            "finite_budget": int(status.get("nonfinite_steps", args.max_nonfinite_steps + 1)) <= args.max_nonfinite_steps,
            "variant_diagnostics": (
                bool(report.get("correction_magnitude_by_baseline_error")) if variant == "R1" else
                bool(report.get("gate_calibration")) if variant == "R2" else
                report.get("params_added", math.inf) < r1_params and report.get("head_conv_flops_per_image", 0) > 0 if variant == "R3" else
                len(report.get("ode_steps", [])) == 3 and report.get("ode_field_evaluations") == 6
            ),
        })
    scores = sorted(reports.items(), key=lambda item: (item[1]["proxy_rbs_vs_baseline"], item[1]["corrupt_epe"], item[1]["clean_epe"], item[0]))
    top2 = [item[1]["variant"] for item in scores[:2]]
    _atomic_json(OUT / "top2.json", {"top2": top2, "screen_scores": [{"run_id": key, "variant": value["variant"], "proxy_rbs_vs_baseline": value["proxy_rbs_vs_baseline"]} for key, value in scores]})
    for variant in top2:
        run_dir = OUT / "top2_seed1" / f"{variant.lower()}_seed1"
        report = _reuse_complete_run(run_dir, args.steps, args.batch_size, args.eval_samples, [args.native_height, args.native_width], args.precision)
        if report is not None:
            print(f"[{variant}/seed1] reuse complete artifact at {run_dir}", flush=True)
        else:
            report = _train_variant(args, variant, 1, OUT / "top2_seed1", bool(compile_result.get("use_compile")), forward_cache)
        reports[f"{variant.lower()}_seed1"] = report
        _write_registry_csv(registry, reports)
    confirm = _evaluate_confirm(args, top2, cache_module)
    _write_block_review("block3_seed1_confirm", {
        "top2_frozen_before_confirm": confirm.get("selection", {}).get("top2") == top2,
        "confirm200": confirm.get("sample_count") == 200,
        "both_seeds_kept_separate": len(confirm.get("candidates", {})) == 2 * len(top2),
        "complete": confirm.get("complete") is True,
    })
    _write_final_comparison(reports, registry, rdiag, compile_result, top2, confirm)
    required = [
        OUT / "preflight_smoke.json", OUT / "baseline.json", OUT / "rdiag.json", OUT / "artifacts/residual_diag.json", OUT / "r5_status.json",
        OUT / "experiment_registry.json", OUT / "experiment_registry.csv", OUT / "top2.json", OUT / "confirm_selection.json",
        OUT / "confirm_summary.json", OUT / "screen_decision.json", OUT / "final_comparison.md",
        OUT / "cache" / "confirm" / "manifest.json",
        OUT / "block_reviews" / "block0_preflight.json", OUT / "block_reviews" / "block1_baseline_rdiag_cache.json",
        OUT / "block_reviews" / "block3_seed1_confirm.json",
        cache_module.FORWARD_CACHE_DIR / "manifest.json",
    ]
    required.extend(OUT / "block_reviews" / f"block2_{variant.lower()}_seed0.json" for variant in VARIANTS)
    required.extend(OUT / f"{variant}_seed0.json" for variant in VARIANTS)
    required.extend(rdiag_histograms)
    required.extend(OUT / "runs" / f"{variant.lower()}_seed0" / "status.json" for variant in VARIANTS)
    required.extend(OUT / "top2_seed1" / f"{variant.lower()}_seed1" / "status.json" for variant in top2)
    missing = [str(path) for path in required if not path.exists()]
    cache_manifest = json.loads((cache_module.FORWARD_CACHE_DIR / "manifest.json").read_text())
    confirm_cache_manifest = json.loads((OUT / "cache" / "confirm" / "manifest.json").read_text())
    if missing or not cache_manifest.get("complete") or not cache_manifest.get("parity", {}).get("passed") or not confirm_cache_manifest.get("complete") or not confirm_cache_manifest.get("parity", {}).get("passed") or not confirm.get("complete"):
        raise RuntimeError(f"residual completion audit failed: missing={missing}, cache_complete={cache_manifest.get('complete')}, confirm_complete={confirm.get('complete')}")
    _atomic_json(OUT / "completed.json", {"contract_id": CONTRACT_ID, "complete": True, "top2": top2, "reports": list(reports), "confirm_complete": True, "forward_cache_parity": cache_manifest["parity"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "rdiag"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--native-height", type=int, default=540)
    parser.add_argument("--native-width", type=int, default=960)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4, help="production batch is locked at 4")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--compile-benchmark", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-nonfinite-steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.log_every <= 0 or args.batch_size < 0 or args.max_batch_size <= 0 or args.eval_samples <= 0 or args.eval_batch_size <= 0 or args.native_height <= 0 or args.native_width <= 0 or args.num_workers < 0 or args.max_nonfinite_steps < 0:
        parser.error("steps, intervals, eval/native sizes, and max batch must be positive; workers and nonfinite limit may be 0")
    if args.command == "run" and args.batch_size != 4:
        parser.error("the completion-first production contract requires --batch-size 4")
    if args.command == "run" and (args.steps != 2500 or args.eval_every != 500 or args.eval_samples != 64 or args.eval_batch_size != 4 or (args.native_height, args.native_width) != (540, 960) or args.precision != "bf16" or args.num_workers != 0 or args.lr != 1e-4 or args.weight_decay != 1e-5 or args.max_nonfinite_steps != 10 or args.compile_benchmark):
        parser.error("production contract is fixed to 2500/500 steps, dev64, eval batch 4, 540x960 BF16, worker 0, AdamW 1e-4/1e-5, nonfinite limit 10, eager mode")
    if args.command == "run":
        run_all(args)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        device = torch.device(args.device)
        predictor, model = _load_base(device, args.precision)
        result = _run_rdiag(model, device, predictor.amp_dtype, _eval_rows(args.eval_samples), args.native_height, args.native_width, args.eval_batch_size)
        _atomic_json(OUT / "artifacts/residual_diag.json", result)
        _atomic_json(OUT / "rdiag.json", result)


if __name__ == "__main__":
    main()
