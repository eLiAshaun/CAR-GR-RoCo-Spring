#!/usr/bin/env python3
"""Small labeled proxy benchmark aligned with the RoCo-Spring score structure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2 as cv
import numpy as np


HERE = Path(__file__).resolve().parent
STAGE25_DIR = HERE.parent
ROOT = STAGE25_DIR.parents[1]
STAGE25_PATH = STAGE25_DIR / "stage25.py"
STAGE1_PATH = ROOT / "experiments" / "stage1_tournament" / "stage1.py"
RESULTS = HERE / "results"

CORRUPTIONS = (
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "rain",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
)
PRESETS = {"fast": 32, "rank": 64, "confirm": 200}


@lru_cache(maxsize=None)
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


def select_balanced(manifest: list[dict], count: int) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in manifest:
        groups.setdefault((row["scene"], row["direction"], row["side"]), []).append(row)
    scenes = {key[0] for key in groups}
    if not groups or len(groups) != 4 * len(scenes) or count % len(groups):
        raise ValueError("sample count must divide complete scene/direction/side strata")
    per_group = count // len(groups)
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row["sample_id"])
        if len(rows) < per_group:
            raise ValueError(f"not enough samples in stratum {key}")
        indices = np.linspace(0, len(rows) - 1, per_group, dtype=int)
        selected.extend(rows[index] for index in indices)
    return sorted(selected, key=lambda row: row["sample_id"])


def _clip(value: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)


def _corrupt_one(image: np.ndarray, name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    value = image.astype(np.float32) / 255.0
    height, width = image.shape[:2]
    if name == "contrast":
        value = (value - value.mean(axis=(0, 1), keepdims=True)) * 0.45 + value.mean(axis=(0, 1), keepdims=True)
    elif name == "elastic_transform":
        dx = cv.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32), (0, 0), 10) * 18
        dy = cv.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32), (0, 0), 10) * 18
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return cv.remap(image, x + dx, y + dy, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    elif name == "frost":
        frost = cv.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 7)
        frost = (frost - frost.min()) / max(float(np.ptp(frost)), 1e-6)
        tint = np.stack((frost, frost * 0.95, frost * 0.75), axis=-1)
        value = value * 0.65 + tint * 0.45
    elif name == "glass_blur":
        blurred = cv.GaussianBlur(image, (7, 7), 1.4)
        dx = cv.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32), (0, 0), 2) * 2
        dy = cv.GaussianBlur(rng.normal(size=(height, width)).astype(np.float32), (0, 0), 2) * 2
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return cv.GaussianBlur(cv.remap(blurred, x + dx, y + dy, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101), (5, 5), 1)
    elif name == "impulse_noise":
        mask = rng.random((height, width))
        value[mask < 0.06] = 0
        value[mask > 0.94] = 1
    elif name == "pixelate":
        small = cv.resize(image, (max(1, width // 4), max(1, height // 4)), interpolation=cv.INTER_AREA)
        return cv.resize(small, (width, height), interpolation=cv.INTER_NEAREST)
    elif name == "rain":
        layer = np.zeros_like(value)
        for _ in range(max(1, height * width // 5000)):
            x = int(rng.integers(0, width)); y = int(rng.integers(-20, height))
            cv.line(layer, (x, y), (min(width - 1, x + 6), min(height - 1, y + 28)), (0.8, 0.8, 0.8), 1)
        value = value * 0.78 + cv.GaussianBlur(layer, (3, 3), 0.8)
    elif name == "saturate":
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * 0.25, 0, 255)
        return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
    elif name == "snow":
        snow = rng.random((height, width)).astype(np.float32)
        snow = cv.GaussianBlur((snow > 0.985).astype(np.float32), (5, 5), 0.8)
        value = value * 0.72 + snow[..., None] * 1.4
    elif name == "spatter":
        mask = cv.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 5)
        mask = np.clip((mask - 0.52) * 12, 0, 0.65)[..., None]
        mud = np.array([0.12, 0.25, 0.38], dtype=np.float32)
        value = value * (1 - mask) + mud * mask
    elif name == "speckle_noise":
        value += value * rng.normal(0, 0.22, value.shape).astype(np.float32)
    elif name == "zoom_blur":
        accum = value.copy()
        for scale in (1.03, 1.06, 1.09, 1.12):
            resized = cv.resize(value, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
            y0 = (resized.shape[0] - height) // 2; x0 = (resized.shape[1] - width) // 2
            accum += resized[y0:y0 + height, x0:x0 + width]
        value = accum / 5
    else:
        raise ValueError(f"unknown local corruption: {name}")
    return _clip(value)


def corrupt_pair(image1: np.ndarray, image2: np.ndarray, name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if name not in CORRUPTIONS or image1.shape != image2.shape:
        raise ValueError("unknown corruption or mismatched pair")
    stage1 = _load(STAGE1_PATH, "fast_benchmark_stage1")
    if name in stage1.CORRUPTIONS:
        return stage1._corrupt_image(image1, name, seed), stage1._corrupt_image(image2, name, seed)
    return _corrupt_one(image1, name, seed), _corrupt_one(image2, name, seed)


def aggregate_records(records: dict[str, dict]) -> dict:
    totals: dict[str, list[float]] = {}
    for row in records.values():
        total = totals.setdefault(row["condition"], [0.0, 0.0])
        total[0] += float(row["error_sum"])
        total[1] += int(row["valid_count"])
    if "clean" not in totals or len(totals) < 2 or any(count <= 0 for _, count in totals.values()):
        raise ValueError("records require clean, corruption, and valid pixels")
    per_condition = {name: error_sum / count for name, (error_sum, count) in totals.items()}
    robust = statistics.fmean(value for name, value in per_condition.items() if name != "clean")
    return {"clean_epe": per_condition["clean"], "robust_mean_epe": robust, "per_condition": per_condition}


def rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["proxy_rbs"], row["robust_term"], row["clean_term"], row["run_id"]))


def resolve_run(run_id: str) -> tuple[str, Path]:
    pretrained = {
        "raft_anchor": ("raft", ROOT / "experiments/stage1_tournament/checkpoints/raft-sintel-fb44381e.ckpt"),
        "dpflow_pretrained": ("dpflow", ROOT / "experiments/stage1_tournament/checkpoints/dpflow-spring-69bac7fa.ckpt"),
        "waft_pretrained": ("waft_dav2_a2", ROOT / "experiments/stage1_tournament/checkpoints/waft_dav2_a2-spring-04a4560e.ckpt"),
    }
    if run_id in pretrained:
        return pretrained[run_id]
    metadata = STAGE25_DIR / "runs" / run_id / "checkpoint.json"
    if not metadata.exists():
        raise FileNotFoundError(f"unknown run/checkpoint: {run_id}")
    checkpoint = Path(json.loads(metadata.read_text())["weights_checkpoint"])
    model = "waft_dav2_a2" if run_id.startswith("waft") else "dpflow"
    return model, checkpoint


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluate(preset: str, run_id: str) -> dict:
    import torch

    stage1 = _load(STAGE1_PATH, "fast_benchmark_stage1_eval")
    stage25 = _load(STAGE25_PATH, "fast_benchmark_stage25")
    count = PRESETS[preset]
    manifest = (
        stage25.build_spring_manifest(stage25.CONFIRM_SCENES)
        if preset == "confirm"
        else stage25.build_dev_manifest()
    )
    samples = select_balanced(manifest, count)
    model_name, checkpoint = resolve_run(run_id)
    path = RESULTS / preset / f"{run_id}.json"
    report = json.loads(path.read_text()) if path.exists() else {
        "run_id": run_id, "preset": preset, "sample_count": count,
        "checkpoint": str(checkpoint), "ptlflow_model": model_name,
        "conditions": ["clean", *CORRUPTIONS], "records": {}, "complete": False,
    }
    records = report["records"]
    model = None
    for sample in samples:
        image1 = cv.imread(sample["image1"], cv.IMREAD_COLOR)
        image2 = cv.imread(sample["image2"], cv.IMREAD_COLOR)
        if image1 is None or image2 is None:
            raise RuntimeError(f"failed to read {sample['sample_id']}")
        gt, valid = stage1.read_spring_gt(Path(sample["gt"]))
        for condition in ("clean", *CORRUPTIONS):
            key = f"{condition}/{sample['sample_id']}"
            if key in records:
                continue
            pair = (image1, image2) if condition == "clean" else corrupt_pair(
                image1, image2, condition, stable_seed(sample["sample_id"], condition)
            )
            if model is None:
                model = stage25._load_model(model_name, checkpoint)
                stage1._predict(model, *pair)
                torch.cuda.reset_peak_memory_stats()
            prediction, runtime_ms = stage1._predict(model, *pair)
            error = np.linalg.norm(prediction - gt, axis=-1)
            mask = valid & np.isfinite(error)
            records[key] = {
                "sample_id": sample["sample_id"], "scene": sample["scene"],
                "direction": sample["direction"], "side": sample["side"],
                "condition": condition, "error_sum": float(error[mask].sum(dtype=np.float64)),
                "valid_count": int(mask.sum()), "runtime_ms": runtime_ms,
            }
            if len(records) % 20 == 0:
                report["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024**2
                _atomic_json(path, report)
                print(f"[{run_id}/{preset}] {len(records)}/{count * 21}", flush=True)
    report.update(
        complete=True, aggregate=aggregate_records(records), metric_count=count * 21,
        median_runtime_ms=statistics.median(row["runtime_ms"] for row in records.values()),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    _atomic_json(path, report)
    return report


def write_ranking(preset: str) -> list[dict]:
    anchor_path = RESULTS / preset / "raft_anchor.json"
    if not anchor_path.exists() or not json.loads(anchor_path.read_text()).get("complete"):
        raise RuntimeError("RAFT anchor is incomplete")
    anchor = json.loads(anchor_path.read_text())["aggregate"]
    rows = []
    for path in sorted((RESULTS / preset).glob("*.json")):
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not report.get("complete"):
            continue
        aggregate = report["aggregate"]
        clean_term = aggregate["clean_epe"] / anchor["clean_epe"]
        robust_term = aggregate["robust_mean_epe"] / anchor["robust_mean_epe"]
        rows.append({
            "run_id": report["run_id"], "preset": preset,
            "clean_epe": aggregate["clean_epe"], "robust_mean_epe": aggregate["robust_mean_epe"],
            "clean_term": clean_term, "robust_term": robust_term,
            "proxy_rbs": 0.5 * (clean_term + robust_term),
            "median_runtime_ms": report["median_runtime_ms"],
        })
    rows = rank_rows(rows)
    with (RESULTS / preset / "ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _atomic_json(RESULTS / preset / "ranking.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("run_id")
    args = parser.parse_args()
    evaluate(args.preset, "raft_anchor")
    evaluate(args.preset, args.run_id)
    print(json.dumps(write_ranking(args.preset), indent=2))


if __name__ == "__main__":
    main()
