#!/usr/bin/env python3
"""Unified pretrained-model tournament for the RoCo-Spring optical-flow track."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import struct
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPRING_ROOT = ROOT / "data" / "spring"
CHECKPOINT_DIR = HERE / "checkpoints"
PREDICTION_DIR = HERE / "predictions"
RESULT_DIR = HERE / "results"
MANIFEST_PATH = HERE / "manifest.json"
SCENE = "0022"
FLOW_MAGIC = 202021.25

CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "gaussian_blur",
    "defocus_blur",
    "motion_blur",
    "brightness",
    "jpeg_compression",
    "fog",
)
CONDITIONS = ("clean",) + CORRUPTIONS

MODELS = {
    "raft": {
        "ptlflow_model": "raft",
        "checkpoint": "sintel",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/raft-sintel-fb44381e.ckpt",
    },
    "sea_raft": {
        "ptlflow_model": "sea_raft_m",
        "checkpoint": "spring",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/sea_raft_m-spring-de7c13e2.ckpt",
    },
    "waft": {
        "ptlflow_model": "waft_dav2_a2",
        "checkpoint": "spring",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/waft_dav2_a2-spring-04a4560e.ckpt",
    },
    "gmflow": {
        "ptlflow_model": "gmflow_refine",
        "checkpoint": "sintel",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/gmflow_refine-sintel-ee46a2c4.ckpt",
    },
    "dpflow": {
        "ptlflow_model": "dpflow",
        "checkpoint": "spring",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-spring-69bac7fa.ckpt",
    },
    "ms_raft_p": {
        "ptlflow_model": "ms_raft_p",
        "checkpoint": "mixed",
        "url": "https://github.com/hmorimitsu/ptlflow/releases/download/weights1/ms_raft_plus-mixed-2bb01f62.ckpt",
    },
}


class Sample(NamedTuple):
    sample_id: str
    scene: str
    side: str
    image1: Path
    image2: Path
    gt: Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_manifest(root: Path = SPRING_ROOT) -> list[Sample]:
    scene_dir = Path(root) / "train" / SCENE
    samples: list[Sample] = []
    for side in ("left", "right"):
        frames = sorted((scene_dir / f"frame_{side}").glob("*.png"))
        flows = sorted((scene_dir / f"flow_FW_{side}").glob("*.flo5"))
        if len(frames) != 19 or len(flows) != 18:
            raise RuntimeError(
                f"Spring {SCENE}/{side} must contain 19 frames and 18 FW flows; "
                f"found {len(frames)} and {len(flows)}"
            )
        for index, (image1, image2, gt) in enumerate(
            zip(frames[:-1], frames[1:], flows), start=1
        ):
            samples.append(
                Sample(
                    f"{SCENE}_{side}_{index:04d}",
                    SCENE,
                    side,
                    image1,
                    image2,
                    gt,
                )
            )
    if len(samples) != 36 or len({sample.sample_id for sample in samples}) != 36:
        raise RuntimeError("Spring Stage 1 manifest must contain 36 unique samples")
    return samples


def write_manifest(samples: list[Sample], path: Path = MANIFEST_PATH) -> None:
    atomic_json(
        path,
        {
            "scene": SCENE,
            "directions": ["FW_left", "FW_right"],
            "count": len(samples),
            "samples": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in sample._asdict().items()
                }
                for sample in samples
            ],
        },
    )


def read_spring_gt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        if "flow" not in handle:
            raise ValueError(f"missing flow dataset in {path}")
        # Read compressed HDF5 chunks contiguously before subsampling.  A
        # strided HDF5 read amplifies NFS chunk I/O while producing the same
        # values several times more slowly.
        flow = np.asarray(handle["flow"][...])[::2, ::2].astype(np.float32, copy=True)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"invalid Spring flow shape {flow.shape} in {path}")
    valid = np.isfinite(flow).all(axis=2) & (np.abs(flow) < 10000).all(axis=2)
    return flow, valid


def write_flo(path: Path, flow: np.ndarray) -> None:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow must be finite HxWx2 float data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    height, width = flow.shape[:2]
    with temporary.open("wb") as handle:
        handle.write(struct.pack("<fii", FLOW_MAGIC, width, height))
        flow.astype("<f4", copy=False).tofile(handle)
    os.replace(temporary, path)


def read_flo(path: Path) -> np.ndarray:
    with Path(path).open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated .flo header: {path}")
        magic, width, height = struct.unpack("<fii", header)
        if magic != FLOW_MAGIC or width <= 0 or height <= 0:
            raise ValueError(f"invalid .flo header: {path}")
        values = np.fromfile(handle, dtype="<f4")
    if values.size != height * width * 2:
        raise ValueError(f"truncated .flo payload: {path}")
    return values.reshape(height, width, 2)


def valid_prediction(path: Path, expected_hw: tuple[int, int]) -> bool:
    try:
        flow = read_flo(path)
        return flow.shape[:2] == tuple(expected_hw) and np.isfinite(flow).all()
    except (OSError, ValueError):
        return False


def flow_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    if prediction.shape != ground_truth.shape or valid.shape != ground_truth.shape[:2]:
        raise ValueError("prediction, ground truth, and valid mask shapes do not match")
    mask = valid & np.isfinite(prediction).all(axis=2)
    if not mask.any():
        raise ValueError("sample has no valid pixels")
    error = np.linalg.norm(prediction - ground_truth, axis=2)
    motion = np.linalg.norm(ground_truth, axis=2)

    def mean_for(extra: np.ndarray) -> float:
        selected = mask & extra
        return float(error[selected].mean()) if selected.any() else math.nan

    return {
        "epe": float(error[mask].mean()),
        "motion_lt10_epe": mean_for(motion < 10),
        "motion_10_40_epe": mean_for((motion >= 10) & (motion < 40)),
        "motion_ge40_epe": mean_for(motion >= 40),
    }


def _disk_kernel(radius: int) -> np.ndarray:
    import cv2 as cv

    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cv.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel / kernel.sum()


def _corrupt_image(image: np.ndarray, name: str, seed: int) -> np.ndarray:
    import cv2 as cv

    if name not in CORRUPTIONS:
        raise ValueError(f"unknown corruption: {name}")
    rng = np.random.default_rng(seed)
    value = image.astype(np.float32) / 255.0
    if name == "gaussian_noise":
        value += rng.normal(0, 0.12, value.shape).astype(np.float32)
    elif name == "shot_noise":
        value = rng.poisson(np.clip(value, 0, 1) * 12.0).astype(np.float32) / 12.0
    elif name == "gaussian_blur":
        value = cv.GaussianBlur(value, (9, 9), 2.0)
    elif name == "defocus_blur":
        value = cv.filter2D(value, -1, _disk_kernel(5))
    elif name == "motion_blur":
        length = 15
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        angle = float(rng.uniform(-45, 45))
        matrix = cv.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
        kernel = cv.warpAffine(kernel, matrix, (length, length))
        kernel /= kernel.sum()
        value = cv.filter2D(value, -1, kernel)
    elif name == "brightness":
        value += 0.18
    elif name == "jpeg_compression":
        ok, encoded = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, 25])
        if not ok:
            raise RuntimeError("OpenCV JPEG encoding failed")
        decoded = cv.imdecode(encoded, cv.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV JPEG decoding failed")
        return decoded
    else:
        coarse = rng.random((max(2, image.shape[0] // 16), max(2, image.shape[1] // 16))).astype(
            np.float32
        )
        coarse = cv.GaussianBlur(coarse, (0, 0), 1.5)
        fog = cv.resize(coarse, (image.shape[1], image.shape[0]), interpolation=cv.INTER_CUBIC)
        fog = (fog - fog.min()) / max(float(fog.max() - fog.min()), 1e-6)
        value = value * 0.55 + fog[..., None] * 0.45
    return np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)


def corrupt_pair(
    image1: np.ndarray, image2: np.ndarray, name: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if image1.shape != image2.shape:
        raise ValueError("corruption requires same-shaped image pairs")
    return _corrupt_image(image1, name, seed), _corrupt_image(image2, name, seed)


def weighted_flow(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    if a.shape != b.shape or not 0 <= alpha <= 1:
        raise ValueError("flows must have equal shapes and alpha must be in [0, 1]")
    return (alpha * a + (1.0 - alpha) * b).astype(np.float32)


def pixel_oracle(flows: list[np.ndarray], ground_truth: np.ndarray) -> np.ndarray:
    if len(flows) < 2 or any(flow.shape != ground_truth.shape for flow in flows):
        raise ValueError("pixel oracle requires at least two shape-compatible flows")
    errors = np.stack([np.linalg.norm(flow - ground_truth, axis=2) for flow in flows])
    best = errors.argmin(axis=0)
    result = np.empty_like(ground_truth)
    for index, flow in enumerate(flows):
        result[best == index] = flow[best == index]
    return result


def family_oracle(values: dict[str, dict[str, float]]) -> dict[str, tuple[str, float]]:
    if not values or any(not model_values for model_values in values.values()):
        raise ValueError("family oracle requires model values for every family")
    return {
        family: min(model_values.items(), key=lambda item: (item[1], item[0]))
        for family, model_values in values.items()
    }


def add_proxy_terms(
    rows: list[dict[str, object]], raft_clean: float, raft_robust: float
) -> list[dict[str, object]]:
    if raft_clean <= 0 or raft_robust <= 0:
        raise ValueError("RAFT normalization errors must be positive")
    enriched = []
    for original in rows:
        row = dict(original)
        spring = float(row["clean_epe"]) / raft_clean
        robust = float(row["corrupted_mean_epe"]) / raft_robust
        row.update(
            spring_normalized=spring,
            robust_normalized=robust,
            proxy_rbs=0.5 * (spring + robust),
        )
        enriched.append(row)
    return sorted(
        enriched,
        key=lambda row: (
            float(row["proxy_rbs"]),
            float(row["robust_normalized"]),
            float(row["spring_normalized"]),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(alias: str) -> Path:
    return CHECKPOINT_DIR / MODELS[alias]["url"].rsplit("/", 1)[-1]


def checkpoint_download_command(url: str, partial: Path) -> list[str]:
    return [
        "aria2c",
        "--continue=true",
        "-x",
        "4",
        "-s",
        "4",
        "-k",
        "1M",
        "--connect-timeout=30",
        "--timeout=120",
        "--max-tries=20",
        "--retry-wait=3",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--summary-interval=30",
        "--console-log-level=notice",
        "--dir",
        str(partial.parent),
        "--out",
        partial.name,
        url,
    ]


def prepare() -> None:
    import ptlflow

    if ptlflow.__version__ != "0.4.2":
        raise RuntimeError(f"expected PTLFlow 0.4.2, found {ptlflow.__version__}")
    missing = {
        spec["ptlflow_model"] for spec in MODELS.values()
    } - set(ptlflow.get_model_names())
    if missing:
        raise RuntimeError(f"PTLFlow is missing required models: {sorted(missing)}")

    samples = build_manifest()
    write_manifest(samples)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = []
    for alias, spec in MODELS.items():
        path = _checkpoint_path(alias)
        partial = path.with_suffix(path.suffix + ".partial")
        if not path.exists():
            subprocess.run(checkpoint_download_command(spec["url"], partial), check=True)
            digest = _sha256(partial)
            expected = re.search(r"-([0-9a-f]{8})\.ckpt$", path.name)
            if expected and not digest.startswith(expected.group(1)):
                raise RuntimeError(f"checkpoint hash mismatch: {partial}")
            os.replace(partial, path)
        digest = _sha256(path)
        expected = re.search(r"-([0-9a-f]{8})\.ckpt$", path.name)
        if expected and not digest.startswith(expected.group(1)):
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        checkpoint_rows.append(
            {
                "model": alias,
                "ptlflow_model": spec["ptlflow_model"],
                "checkpoint_id": spec["checkpoint"],
                "url": spec["url"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    _write_csv(RESULT_DIR / "checkpoint_manifest.csv", checkpoint_rows)
    print(f"prepared {len(samples)} samples and {len(checkpoint_rows)} checkpoints")


def _stable_seed(sample_id: str, condition: str) -> int:
    digest = hashlib.sha256(f"{sample_id}:{condition}:severity3".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _load_model(alias: str):
    import ptlflow

    path = _checkpoint_path(alias)
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint; run setup first: {path}")
    spec = MODELS[alias]
    model = ptlflow.get_model(spec["ptlflow_model"], ckpt_path=str(path))
    return model.cuda().eval()


def _predict(model, image1: np.ndarray, image2: np.ndarray) -> tuple[np.ndarray, float]:
    import torch
    from ptlflow.utils.io_adapter import IOAdapter

    if hasattr(model, "prev_preds"):
        model.prev_preds = None
    adapter = IOAdapter(model.output_stride, image1.shape[:2], cuda=True)
    inputs = adapter.prepare_inputs(images=[image1, image2])
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = adapter.unscale(model(inputs))
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    flow = outputs["flows"][0, 0].detach().float().cpu().permute(1, 2, 0).numpy()
    if flow.shape[:2] != image1.shape[:2] or not np.isfinite(flow).all():
        raise RuntimeError(
            f"invalid model output: expected {image1.shape[:2]}, found {flow.shape}"
        )
    return flow.astype(np.float32, copy=False), elapsed_ms


def _load_run_metadata(path: Path, alias: str) -> dict[str, object]:
    if path.exists():
        return json.loads(path.read_text())
    return {"model": alias, "records": {}, "complete": False, "peak_vram_mb": 0.0}


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else math.nan


def _aggregate_metrics(metrics: list[dict[str, object]]) -> dict[str, float]:
    keys = ("epe", "motion_lt10_epe", "motion_10_40_epe", "motion_ge40_epe")
    return {key: _mean([row[key] for row in metrics]) for key in keys}


def _prediction_namespace(alias: str, smoke: bool) -> Path:
    return Path("_smoke") / alias if smoke else Path(alias)


def run_model(alias: str, smoke: bool = False) -> dict[str, object]:
    import cv2 as cv
    import torch

    if alias not in MODELS:
        raise ValueError(f"unknown model: {alias}")
    samples = build_manifest()
    write_manifest(samples)
    selected_samples = samples[:1] if smoke else samples
    conditions = ("clean", "gaussian_noise") if smoke else CONDITIONS
    namespace = _prediction_namespace(alias, smoke)
    metadata_path = RESULT_DIR / (f"smoke_{alias}.json" if smoke else f"model_{alias}.json")
    metadata = _load_run_metadata(metadata_path, alias)
    records = metadata.setdefault("records", {})
    model = None
    warmed_up = False
    peak_vram = float(metadata.get("peak_vram_mb", 0.0))

    for sample in selected_samples:
        image1 = cv.imread(str(sample.image1), cv.IMREAD_COLOR)
        image2 = cv.imread(str(sample.image2), cv.IMREAD_COLOR)
        if image1 is None or image2 is None:
            raise RuntimeError(f"failed to read images for {sample.sample_id}")
        gt, valid = read_spring_gt(sample.gt)
        if gt.shape[:2] != image1.shape[:2]:
            raise RuntimeError(
                f"2K GT/image mismatch for {sample.sample_id}: {gt.shape} vs {image1.shape}"
            )

        for condition in conditions:
            key = f"{condition}/{sample.sample_id}"
            prediction_path = PREDICTION_DIR / namespace / condition / f"{sample.sample_id}.flo"
            if valid_prediction(prediction_path, gt.shape[:2]):
                prediction = read_flo(prediction_path)
                elapsed_ms = records.get(key, {}).get("runtime_ms")
            else:
                if condition == "clean":
                    pair = image1, image2
                else:
                    pair = corrupt_pair(
                        image1, image2, condition, _stable_seed(sample.sample_id, condition)
                    )
                if model is None:
                    model = _load_model(alias)
                if not warmed_up:
                    for _ in range(1 if smoke else 2):
                        _predict(model, *pair)
                    torch.cuda.reset_peak_memory_stats()
                    warmed_up = True
                prediction, elapsed_ms = _predict(model, *pair)
                write_flo(prediction_path, prediction)
                peak_vram = max(
                    peak_vram, torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                )

            metrics = flow_metrics(prediction, gt, valid)
            records[key] = {
                "condition": condition,
                "sample_id": sample.sample_id,
                "prediction": str(prediction_path),
                "runtime_ms": elapsed_ms,
                **metrics,
            }
            metadata.update(
                ptlflow_model=MODELS[alias]["ptlflow_model"],
                checkpoint_id=MODELS[alias]["checkpoint"],
                checkpoint_path=str(_checkpoint_path(alias)),
                input_size=[int(image1.shape[0]), int(image1.shape[1])],
                resize="none",
                padding="model_internal_to_output_stride",
                model_output_stride=getattr(model, "output_stride", None),
                peak_vram_mb=peak_vram,
                complete=False,
            )
            atomic_json(metadata_path, metadata)
            print(f"[{alias}] {condition} {sample.sample_id}: EPE={metrics['epe']:.4f}", flush=True)

    expected = len(selected_samples) * len(conditions)
    complete_records = [
        row
        for row in records.values()
        if row.get("condition") in conditions
        and row.get("sample_id") in {sample.sample_id for sample in selected_samples}
    ]
    if len(complete_records) != expected:
        raise RuntimeError(f"expected {expected} records, found {len(complete_records)}")
    per_condition = {
        condition: _aggregate_metrics(
            [row for row in complete_records if row["condition"] == condition]
        )
        for condition in conditions
    }
    runtimes = [
        float(row["runtime_ms"])
        for row in complete_records
        if row.get("runtime_ms") is not None
    ]
    metadata.update(
        complete=True,
        prediction_count=expected,
        per_condition=per_condition,
        median_runtime_ms=statistics.median(runtimes) if runtimes else None,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    atomic_json(metadata_path, metadata)
    if not smoke:
        refresh_summaries()
    return metadata


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _completed_reports(aliases: list[str] | None = None) -> dict[str, dict[str, object]]:
    reports = {}
    for alias in aliases or list(MODELS):
        path = RESULT_DIR / f"model_{alias}.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report.get("complete"):
                reports[alias] = report
    return reports


def refresh_summaries() -> None:
    reports = _completed_reports()
    baseline_rows = []
    corruption_rows = []
    for alias, report in reports.items():
        per_condition = report["per_condition"]
        clean = per_condition["clean"]
        robust = _mean([per_condition[name]["epe"] for name in CORRUPTIONS])
        baseline_rows.append(
            {
                "model": alias,
                "ptlflow_model": report["ptlflow_model"],
                "checkpoint_id": report["checkpoint_id"],
                "clean_epe": clean["epe"],
                "corrupted_mean_epe": robust,
                "motion_lt10_epe": clean["motion_lt10_epe"],
                "motion_10_40_epe": clean["motion_10_40_epe"],
                "motion_ge40_epe": clean["motion_ge40_epe"],
                "median_runtime_ms": report.get("median_runtime_ms"),
                "peak_vram_mb": report.get("peak_vram_mb"),
            }
        )
        for condition in CORRUPTIONS:
            corruption_rows.append(
                {"model": alias, "corruption": condition, **per_condition[condition]}
            )

    if "raft" in reports:
        raft = next(row for row in baseline_rows if row["model"] == "raft")
        baseline_rows = add_proxy_terms(
            baseline_rows,
            float(raft["clean_epe"]),
            float(raft["corrupted_mean_epe"]),
        )
    else:
        baseline_rows.sort(key=lambda row: row["model"])
    _write_csv(RESULT_DIR / "baseline_summary.csv", baseline_rows)
    _write_csv(RESULT_DIR / "per_corruption.csv", corruption_rows)
    if corruption_rows:
        _write_heatmap(reports)


def _write_heatmap(reports: dict[str, dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aliases = list(reports)
    matrix = np.array(
        [
            [reports[alias]["per_condition"][condition]["epe"] for condition in CORRUPTIONS]
            for alias in aliases
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(figsize=(12, max(3, 0.7 * len(aliases))))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=35, ha="right")
    axis.set_yticks(range(len(aliases)), aliases)
    axis.set_title("Stage 1 corruption EPE (lower is better)")
    figure.colorbar(image, ax=axis, label="EPE")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "corruption_heatmap.png", dpi=160)
    plt.close(figure)


def _strategy_summary(
    name: str,
    strategy_type: str,
    per_condition: dict[str, list[dict[str, float]]],
    details: dict[str, object],
) -> dict[str, object]:
    aggregate = {
        condition: _aggregate_metrics(metrics) for condition, metrics in per_condition.items()
    }
    clean = aggregate["clean"]
    return {
        "model": name,
        "type": strategy_type,
        "models": details.get("models", ""),
        "alpha": details.get("alpha", ""),
        "clean_epe": clean["epe"],
        "corrupted_mean_epe": _mean([aggregate[name]["epe"] for name in CORRUPTIONS]),
        "motion_lt10_epe": clean["motion_lt10_epe"],
        "motion_10_40_epe": clean["motion_10_40_epe"],
        "motion_ge40_epe": clean["motion_ge40_epe"],
        "details": json.dumps(details, sort_keys=True),
    }


def combine(aliases: list[str]) -> list[dict[str, object]]:
    if len(aliases) not in (2, 3) or len(set(aliases)) != len(aliases):
        raise ValueError("combine requires two or three unique models")
    if any(alias not in MODELS for alias in aliases):
        raise ValueError("combine received an unknown model")
    reports = _completed_reports(aliases)
    if set(reports) != set(aliases):
        raise RuntimeError("every selected model must have a complete full evaluation")
    raft_reports = _completed_reports(["raft"])
    if "raft" not in raft_reports:
        raise RuntimeError("complete RAFT evaluation is required for normalization")

    samples = build_manifest()
    strategies: dict[str, dict[str, object]] = {}
    for a, b in itertools.combinations(aliases, 2):
        for alpha in (0.25, 0.5, 0.75):
            name = f"{a}+{b}@{alpha:.2f}"
            strategies[name] = {
                "type": "simple_average" if alpha == 0.5 else "weighted_average",
                "weights": {a: alpha, b: 1.0 - alpha},
                "details": {"models": f"{a},{b}", "alpha": alpha},
            }
    if len(aliases) == 3:
        name = "+".join(aliases) + "@mean"
        strategies[name] = {
            "type": "three_model_average",
            "weights": {alias: 1.0 / 3.0 for alias in aliases},
            "details": {"models": ",".join(aliases), "alpha": "equal"},
        }

    family_values = {
        condition: {
            alias: float(reports[alias]["per_condition"][condition]["epe"])
            for alias in aliases
        }
        for condition in CONDITIONS
    }
    family_choices = family_oracle(family_values)
    family_name = "per_family_oracle"
    pixel_name = "per_pixel_oracle"
    all_metrics = {
        name: {condition: [] for condition in CONDITIONS}
        for name in (*strategies, family_name, pixel_name)
    }

    for sample in samples:
        gt, valid = read_spring_gt(sample.gt)
        for condition in CONDITIONS:
            flows = {
                alias: read_flo(
                    PREDICTION_DIR / alias / condition / f"{sample.sample_id}.flo"
                )
                for alias in aliases
            }
            for name, strategy in strategies.items():
                fused = sum(
                    float(weight) * flows[alias]
                    for alias, weight in strategy["weights"].items()
                ).astype(np.float32)
                all_metrics[name][condition].append(flow_metrics(fused, gt, valid))
            selected = family_choices[condition][0]
            all_metrics[family_name][condition].append(
                flow_metrics(flows[selected], gt, valid)
            )
            all_metrics[pixel_name][condition].append(
                flow_metrics(pixel_oracle(list(flows.values()), gt), gt, valid)
            )

    rows = [
        _strategy_summary(
            name,
            strategy["type"],
            all_metrics[name],
            strategy["details"],
        )
        for name, strategy in strategies.items()
    ]
    rows.append(
        _strategy_summary(
            family_name,
            family_name,
            all_metrics[family_name],
            {
                "models": ",".join(aliases),
                "selected": {condition: choice[0] for condition, choice in family_choices.items()},
            },
        )
    )
    rows.append(
        _strategy_summary(
            pixel_name,
            pixel_name,
            all_metrics[pixel_name],
            {"models": ",".join(aliases)},
        )
    )
    raft = raft_reports["raft"]
    raft_clean = float(raft["per_condition"]["clean"]["epe"])
    raft_robust = _mean(
        [raft["per_condition"][condition]["epe"] for condition in CORRUPTIONS]
    )
    rows = add_proxy_terms(rows, raft_clean, raft_robust)
    _write_csv(RESULT_DIR / "combination_summary.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("model", choices=MODELS)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("models", nargs="*", default=["raft", "sea_raft"])
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("models", nargs="+", choices=MODELS)
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "model":
        run_model(args.model)
    elif args.command == "smoke":
        for alias in args.models:
            run_model(alias, smoke=True)
    else:
        combine(args.models)


if __name__ == "__main__":
    main()
