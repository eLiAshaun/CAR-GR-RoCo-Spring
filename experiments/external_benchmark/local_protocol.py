"""Fixed local-only Spring/RobustSpring evaluation protocol and metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
DEV_SCENES = ("0002", "0006", "0015", "0022")
CONFIRM_SCENES = ("0004", "0009", "0012", "0023", "0025")
ROBUST_CORRUPTIONS = (
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "rain",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
)
PROTOCOL_VERSION = "local-v1-confirm-full-robust-balanced-200"


def _sorted_frames(scene_dir: Path, side: str) -> list[Path]:
    frames = sorted((scene_dir / f"frame_{side}").glob("*.png"))
    if not frames:
        raise FileNotFoundError(scene_dir / f"frame_{side}")
    return frames


def _frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def build_labeled_manifest(
    scenes: tuple[str, ...] | list[str], data_root: Path = DATA_ROOT
) -> list[dict]:
    """Build complete public Spring train pairs with public flow GT."""
    spring_root = Path(data_root) / "spring" / "train"
    samples: list[dict] = []
    for scene in scenes:
        scene_dir = spring_root / scene
        for side in ("left", "right"):
            frames_ascending = _sorted_frames(scene_dir, side)
            for direction in ("FW", "BW"):
                reverse = direction == "BW"
                frames = list(reversed(frames_ascending)) if reverse else frames_ascending
                flow_dir = scene_dir / f"flow_{direction}_{side}"
                flows = sorted(flow_dir.glob("*.flo5"), reverse=reverse)
                if len(frames) != len(flows) + 1:
                    raise RuntimeError(
                        f"unaligned Spring {scene}/{direction}_{side}: "
                        f"{len(frames)} frames, {len(flows)} flows"
                    )
                for image1, image2, gt in zip(frames[:-1], frames[1:], flows):
                    index = _frame_index(gt)
                    samples.append(
                        {
                            "sample_id": f"{scene}_{direction}_{side}_{index:04d}",
                            "scene": scene,
                            "direction": direction,
                            "side": side,
                            "image1": str(image1),
                            "image2": str(image2),
                            "gt": str(gt),
                        }
                    )
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise RuntimeError("labeled manifest contains duplicate sample IDs")
    return sorted(samples, key=lambda row: row["sample_id"])


def build_test_manifest(data_root: Path = DATA_ROOT) -> list[dict]:
    """Build all clean public test pairs; GT is intentionally absent."""
    test_root = Path(data_root) / "spring" / "test"
    samples: list[dict] = []
    for scene_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for side in ("left", "right"):
            frames_ascending = _sorted_frames(scene_dir, side)
            for direction in ("FW", "BW"):
                reverse = direction == "BW"
                frames = list(reversed(frames_ascending)) if reverse else frames_ascending
                for image1, image2 in zip(frames[:-1], frames[1:]):
                    index = _frame_index(image1)
                    samples.append(
                        {
                            "sample_id": f"{scene_dir.name}_{direction}_{side}_{index:04d}",
                            "scene": scene_dir.name,
                            "direction": direction,
                            "side": side,
                            "image1": str(image1),
                            "image2": str(image2),
                            "relative_image1": str(image1.relative_to(test_root)),
                            "relative_image2": str(image2.relative_to(test_root)),
                        }
                    )
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise RuntimeError("test manifest contains duplicate sample IDs")
    return sorted(samples, key=lambda row: row["sample_id"])


def select_balanced_test(rows: list[dict], count: int) -> list[dict]:
    """Select fixed evenly spaced samples from every scene/direction/side stratum."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["scene"], row["direction"], row["side"])
        groups.setdefault(key, []).append(row)
    if not groups or count <= 0 or count % len(groups):
        raise ValueError("sample count must divide all complete test strata")
    per_group = count // len(groups)
    selected: list[dict] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["sample_id"])
        if len(group) < per_group:
            raise ValueError(f"not enough samples in stratum {key}")
        indices = np.linspace(0, len(group) - 1, per_group, dtype=int)
        selected.extend(group[index] for index in indices)
    return sorted(selected, key=lambda row: row["sample_id"])


def _validate_pair(prediction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim < 2 or prediction.shape[-1] != 2:
        raise ValueError("flow pairs must have the same shape (..., 2)")
    valid = np.isfinite(prediction).all(axis=-1) & np.isfinite(reference).all(axis=-1)
    if not valid.any():
        raise ValueError("flow pair has no finite pixels")
    return prediction, reference


def _statistics(error: np.ndarray, reference_norm: np.ndarray, valid: np.ndarray, kind: str) -> dict:
    error = np.asarray(error, dtype=np.float64)
    reference_norm = np.asarray(reference_norm, dtype=np.float64)
    values = error[valid]
    ref_values = reference_norm[valid]
    thresholds = np.arange(1, 101, dtype=np.float64) / 20.0
    # One sort replaces 100 full-array scans while preserving values <= threshold.
    sorted_values = np.sort(values)
    return {
        "kind": kind,
        "valid_count": int(values.size),
        "epe_sum": float(values.sum(dtype=np.float64)),
        "one_px_count": int(np.count_nonzero(values > 1.0)),
        "three_px_count": int(np.count_nonzero(values > 3.0)),
        "fl_count": int(np.count_nonzero((values > 3.0) & (values > 0.05 * ref_values))),
        "wauc_counts": np.searchsorted(sorted_values, thresholds, side="right").astype(np.int64).tolist(),
    }


def accuracy_sufficient_statistics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict:
    prediction, ground_truth = _validate_pair(prediction, ground_truth)
    finite = np.isfinite(ground_truth).all(axis=-1)
    if valid is None:
        valid = finite
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != finite.shape:
            raise ValueError("valid mask must match flow height/width")
        valid = valid & finite
    error = np.linalg.norm(prediction - ground_truth, axis=-1)
    reference_norm = np.linalg.norm(ground_truth, axis=-1)
    return _statistics(error, reference_norm, valid, "accuracy")


def robustness_sufficient_statistics(clean_prediction: np.ndarray, corrupted_prediction: np.ndarray) -> dict:
    corrupted_prediction, clean_prediction = _validate_pair(corrupted_prediction, clean_prediction)
    error = np.linalg.norm(corrupted_prediction - clean_prediction, axis=-1)
    reference_norm = np.linalg.norm(clean_prediction, axis=-1)
    valid = np.isfinite(clean_prediction).all(axis=-1) & np.isfinite(corrupted_prediction).all(axis=-1)
    return _statistics(error, reference_norm, valid, "robustness")


def merge_statistics(statistics: list[dict]) -> dict:
    if not statistics:
        raise ValueError("cannot merge empty statistics")
    kinds = {item["kind"] for item in statistics}
    if len(kinds) != 1:
        raise ValueError("cannot merge different metric kinds")
    if any(len(item["wauc_counts"]) != 100 for item in statistics):
        raise ValueError("WAUC sufficient statistics must have 100 thresholds")
    return {
        "kind": statistics[0]["kind"],
        "valid_count": sum(int(item["valid_count"]) for item in statistics),
        "epe_sum": sum(float(item["epe_sum"]) for item in statistics),
        "one_px_count": sum(int(item["one_px_count"]) for item in statistics),
        "three_px_count": sum(int(item["three_px_count"]) for item in statistics),
        "fl_count": sum(int(item["fl_count"]) for item in statistics),
        "wauc_counts": [
            sum(int(item["wauc_counts"][index]) for item in statistics)
            for index in range(100)
        ],
    }


def summarize_statistics(statistics: dict) -> dict[str, float | int]:
    count = int(statistics["valid_count"])
    if count <= 0:
        raise ValueError("statistics contain no valid pixels")
    weights = 1.0 - np.arange(100, dtype=np.float64) / 100.0
    wauc_counts = np.asarray(statistics["wauc_counts"], dtype=np.float64)
    wauc = 100.0 * float(np.dot(weights, wauc_counts)) / (count * float(weights.sum()))
    prefix = "delta_" if statistics["kind"] == "robustness" else ""
    return {
        "valid_count": count,
        f"{prefix}epe": float(statistics["epe_sum"]) / count,
        f"{prefix}one_px": 100.0 * int(statistics["one_px_count"]) / count,
        f"{prefix}three_px": 100.0 * int(statistics["three_px_count"]) / count,
        f"{prefix}fl": 100.0 * int(statistics["fl_count"]) / count,
        f"{prefix}wauc": wauc,
    }


def group_statistics(records: list[dict], group_key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record[group_key]), []).append(record["statistics"])
    return {
        key: summarize_statistics(merge_statistics(values))
        for key, values in sorted(grouped.items())
    }


def manifest_digest(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
