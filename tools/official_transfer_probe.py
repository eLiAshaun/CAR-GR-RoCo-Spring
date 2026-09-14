"""Run and audit the frozen CAR-GR-WAFT official transfer probe.

This runner reuses the existing WAFT model, CAR-GR residual head, and native
540x960 forward helpers. It only adds test-set I/O and output validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/external_benchmark/sources/WAFT"
sys.path[:0] = [str(ROOT), str(SOURCE)]

from experiments.residual_screen import residual_experiment as R
import train_stage34 as S
from utils import frame_utils


PROBE = ROOT / "results/car_vs_cargr/official_transfer_probe"
DATA = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs/car_gr_step2000_official"
CHECKPOINT = ROOT / "results/car_vs_cargr/CAR_GR/best.pt"
CONFIG = R.CONFIG
NATIVE_SIZE = (540, 960)
IMAGE_SIZE = (1080, 1920)
CHECKPOINT_STEP = 2000
TRAINING_LR = 2.5e-5
FLOW_EXTREME_THRESHOLD = 40_000.0
CAMERAS = ("left", "right")
DIRECTIONS = ("FW", "BW")


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SOURCE), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def logger_for(name: str, path: Path, resume: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="a" if resume else "w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def frame_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.png"))
    numbers = []
    for path in files:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"invalid frame filename: {path}") from error
    if numbers != list(range(1, len(files) + 1)):
        raise RuntimeError(f"non-contiguous frames in {directory}: {numbers[:5]} ...")
    return files


def scan_data() -> dict:
    clean_root = DATA / "spring" / "test"
    robust_root = DATA / "robust_spring"
    clean_scenes = sorted(path.name for path in clean_root.iterdir() if path.is_dir())
    corruptions = sorted(path.name for path in robust_root.iterdir() if path.is_dir())
    if not clean_scenes or len(corruptions) != 20:
        raise RuntimeError(
            f"unexpected test inventory: scenes={clean_scenes}, corruptions={corruptions}"
        )

    scene_frames = {}
    image_shapes = set()
    for scene in clean_scenes:
        scene_frames[scene] = {}
        for side in CAMERAS:
            files = frame_files(clean_root / scene / f"frame_{side}")
            if not files:
                raise RuntimeError(f"no clean frames: {scene}/{side}")
            image = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read {files[0]}")
            image_shapes.add(tuple(image.shape))
            if tuple(image.shape) != IMAGE_SIZE + (3,):
                raise RuntimeError(f"unexpected image shape {files[0]}: {image.shape}")
            scene_frames[scene][side] = len(files)

    for corruption in corruptions:
        root = robust_root / corruption / "test"
        scenes = sorted(path.name for path in root.iterdir() if path.is_dir())
        if scenes != clean_scenes:
            raise RuntimeError(f"scene mismatch for {corruption}: {scenes}")
        for scene in clean_scenes:
            for side in CAMERAS:
                files = frame_files(root / scene / f"frame_{side}")
                if len(files) != scene_frames[scene][side]:
                    raise RuntimeError(
                        f"frame count mismatch for {corruption}/{scene}/{side}"
                    )
                image = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
                if image is None or tuple(image.shape) != IMAGE_SIZE + (3,):
                    raise RuntimeError(
                        f"unexpected robust image shape {root / scene / f'frame_{side}'}"
                    )

    pairs_per_clean = 4 * sum(
        values["left"] - 1 for values in scene_frames.values()
    )
    return {
        "clean_root": str(clean_root.resolve()),
        "robust_root": str(robust_root.resolve()),
        "clean_scenes": clean_scenes,
        "robust_corruptions": corruptions,
        "scene_frame_counts": scene_frames,
        "image_shapes": [list(shape) for shape in sorted(image_shapes)],
        "clean_prediction_count": pairs_per_clean,
        "robust_prediction_count": pairs_per_clean * len(corruptions),
    }


def output_path(
    stage: str,
    condition: str | None,
    scene: str,
    side: str,
    direction: str,
    frame: int,
) -> Path:
    root = OUTPUT_ROOT / "spring-robust"
    if stage == "clean":
        root /= "clean"
        root /= "test"
    else:
        if condition is None:
            raise ValueError("robust job needs a corruption name")
        root /= condition
        root /= "test"
    directory = root / scene / f"flow_{direction}_{side}"
    return directory / f"flow_{direction}_{side}_{frame:04d}.flo5"


def jobs_for(stage: str, inventory: dict) -> list[dict]:
    jobs = []
    conditions = [None] if stage == "clean" else inventory["robust_corruptions"]
    for condition in conditions:
        for scene in inventory["clean_scenes"]:
            frame_count = inventory["scene_frame_counts"][scene]["left"]
            for side in CAMERAS:
                frame_dir = DATA / "spring" / "test" / scene / f"frame_{side}"
                if condition is not None:
                    frame_dir = (
                        DATA
                        / "robust_spring"
                        / condition
                        / "test"
                        / scene
                        / f"frame_{side}"
                    )
                files = frame_files(frame_dir)
                for direction in DIRECTIONS:
                    frames = (
                        range(1, frame_count)
                        if direction == "FW"
                        else range(frame_count, 1, -1)
                    )
                    for frame in frames:
                        second = frame + 1 if direction == "FW" else frame - 1
                        jobs.append(
                            {
                                "stage": stage,
                                "condition": condition,
                                "scene": scene,
                                "side": side,
                                "direction": direction,
                                "frame": frame,
                                "image1": files[frame - 1],
                                "image2": files[second - 1],
                                "output": output_path(
                                    stage, condition, scene, side, direction, frame
                                ),
                            }
                        )
    return jobs


def write_frozen_manifest(inventory: dict) -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    comparison = json.loads((ROOT / "results/car_vs_cargr/comparison.json").read_text())
    winner = comparison["CAR_GR"]
    evaluation = json.loads(
        (ROOT / "results/car_vs_cargr/CAR_GR/evaluations/eval_step_2000.json").read_text()
    )
    if payload.get("branch") != "CAR_GR" or payload.get("step") != CHECKPOINT_STEP:
        raise RuntimeError("winner checkpoint payload is not CAR_GR step 2000")
    if abs(winner["clean_epe"] - 2.02480743780023) > 1e-12:
        raise RuntimeError("comparison winner clean metric does not match handoff")
    if abs(winner["corrupt_epe"] - 2.727980523317088) > 1e-12:
        raise RuntimeError("comparison winner corrupt metric does not match handoff")
    manifest = {
        "branch": "CAR-GR-WAFT",
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint": str(CHECKPOINT.resolve()),
        "training_lr": TRAINING_LR,
        "clean_proxy_epe": 2.02480743780023,
        "corrupt_proxy_epe": 2.727980523317088,
        "proxy_rbs": 0.97168343552713,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_size_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_mtime_utc": iso_mtime(CHECKPOINT),
        "checkpoint_payload": {
            "contract_id": payload.get("contract_id"),
            "branch": payload.get("branch"),
            "step": payload.get("step"),
            "model_tensor_count": len(payload["model"]),
            "head_tensor_count": len(payload["head"]),
        },
        "config": {
            "path": str(CONFIG.resolve()),
            "sha256": sha256(CONFIG),
            "algorithm": "waft-a2",
            "feature_encoder": "dav2",
            "iterative_module": "vits",
            "image_size": list(NATIVE_SIZE),
            "input_image_size": list(IMAGE_SIZE),
            "input_scale": 0.5,
            "precision": "bf16",
            "tf32": True,
            "iters": 5,
            "correlation": "WAFT native fmap warp via coords_grid + bilinear_sampler",
            "padding": "WAFT Padder factor=112; model unpads predictions and auxiliary tensors",
            "flow_resize": "bilinear align_corners=True from 540x960 to 1080x1920; x/y scaled by 2",
            "preprocessing": "cv2 BGR decode -> RGB; float 0..255; bilinear align_corners=True; ImageNet normalize in model",
            "head": "RiskGatedResidual(hidden=64); prediction = base_flow + gated_delta",
            "return_aux": True,
        },
        "source": {
            "path": str(SOURCE.resolve()),
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--short")),
        },
        "data": inventory,
        "outputs": {
            "root": str(OUTPUT_ROOT.resolve()),
            "clean_input": str((OUTPUT_ROOT / "spring-robust/clean/test").resolve()),
            "robust_input": str((OUTPUT_ROOT / "spring-robust").resolve()),
            "clean_expected_predictions": inventory["clean_prediction_count"],
            "robust_expected_predictions": inventory["robust_prediction_count"],
        },
        "official_tools": {
            "archive": str((PROBE / "subsampling_tools.zip").resolve()),
            "flow_subsampling": str(
                (PROBE / "subsampling_tools/flow_subsampling").resolve()
            ),
            "flow_robust_subsampling": str(
                (PROBE / "subsampling_tools/flow_robust_subsampling").resolve()
            ),
        },
        "source_metric_artifacts": {
            "comparison": str((ROOT / "results/car_vs_cargr/comparison.json").resolve()),
            "evaluation": str(
                (ROOT / "results/car_vs_cargr/CAR_GR/evaluations/eval_step_2000.json").resolve()
            ),
            "evaluation_step": evaluation.get("step"),
            "evaluation_contract_id": evaluation.get("contract_id"),
        },
    }
    json_write(PROBE / "frozen_model_manifest.json", manifest)
    (PROBE / "changes.md").write_text(
        "# Official transfer probe changes\n\n"
        "- Added tools/official_transfer_probe.py as an isolated test-set runner.\n"
        "- I/O only: it maps clean and RobustSpring image pairs to the official "
        "flow_*.flo5 directory layout and uses the existing WAFT .flo5 writer.\n"
        "- The model, checkpoint tensors, optimizer state, iterations, preprocessing, "
        "padding, flow scaling, correlation/warp path, and CAR-GR head are unchanged.\n"
        "- HDF5 generation is delegated to the downloaded official flow_subsampling "
        "and flow_robust_subsampling executables.\n"
        "- No training, fine-tuning, checkpoint averaging, augmentation, GT, or web submission.\n"
    )


def load_frozen_model() -> tuple[torch.nn.Module, torch.nn.Module, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("official transfer probe requires CUDA")
    args = S.load_config(CONFIG)
    args.algorithm = "waft-a2"
    args.image_size = list(NATIVE_SIZE)
    args.precision = "bf16"
    args.device = "cuda"
    model = S.build_model(args)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    head = R.RiskGatedResidual(hidden=64)
    head.load_state_dict(payload["head"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    head.to(device).eval()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return model, head, device


def write_prediction(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame_utils.writeFlo5File(flow.astype(np.float32, copy=False), str(temporary))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_stage(
    stage: str,
    jobs: list[dict],
    model: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    resume: bool = False,
    writers: int = 4,
) -> None:
    log = logger_for(stage, PROBE / f"{stage}_inference.log", resume)
    if resume:
        jobs = [job for job in jobs if not job["output"].exists()]
    log.info("stage=%s jobs=%d batch_size=%d writers=%d resume=%s", stage, len(jobs), batch_size, writers, resume)
    started = time.perf_counter()
    pending = []
    # ponytail: h5py serializes gzip work across threads; processes are the
    # smallest change that actually uses the available CPU for .flo5 writes.
    with ProcessPoolExecutor(
        max_workers=writers, mp_context=mp.get_context("spawn")
    ) as writer_pool:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            images1, images2 = [], []
            for job in batch:
                image1 = cv2.imread(str(job["image1"]), cv2.IMREAD_COLOR)
                image2 = cv2.imread(str(job["image2"]), cv2.IMREAD_COLOR)
                if image1 is None or image2 is None:
                    raise RuntimeError(f"failed to read pair: {job['image1']} {job['image2']}")
                if image1.shape != IMAGE_SIZE + (3,) or image2.shape != IMAGE_SIZE + (3,):
                    raise RuntimeError(f"unexpected pair shape in {job['image1']}")
                images1.append(image1)
                images2.append(image2)
            native1 = R._native_batch(images1, device, *NATIVE_SIZE)
            native2 = R._native_batch(images2, device, *NATIVE_SIZE)
            with torch.no_grad():
                bundle = R._forward_bundle(model, native1, native2, device, torch.bfloat16)
                output = head(bundle)
                prediction = bundle["f0"] + output["delta"]
                flows = R._full_flow(prediction, *IMAGE_SIZE, *NATIVE_SIZE)
            for job, flow in zip(batch, flows):
                if flow.shape != IMAGE_SIZE + (2,):
                    raise RuntimeError(f"unexpected prediction shape for {job['output']}: {flow.shape}")
                if not np.isfinite(flow).all():
                    raise FloatingPointError(f"non-finite prediction for {job['output']}")
                pending.append(writer_pool.submit(write_prediction, job["output"], flow))
            while len(pending) > writers * 2:
                pending.pop(0).result()
            done = start + len(batch)
            if done == len(jobs) or done % (batch_size * 25) == 0:
                elapsed = time.perf_counter() - started
                log.info("processed=%d/%d rate=%.2f pairs/s", done, len(jobs), done / max(elapsed, 1e-9))
        for future in pending:
            future.result()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    log.info("completed stage=%s elapsed_s=%.3f", stage, time.perf_counter() - started)


def validate_predictions(stage: str, jobs: list[dict]) -> dict:
    expected = {job["output"] for job in jobs}
    root = OUTPUT_ROOT / "spring-robust"
    if stage == "clean":
        actual = set((root / "clean" / "test").rglob("*.flo5")) if root.exists() else set()
    else:
        actual = {
            path for path in root.rglob("*.flo5") if "clean" not in path.parts
        } if root.exists() else set()
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    shape_counts = {}
    dtype_counts = {}
    nan_count = inf_count = extreme_count = 0
    max_abs = 0.0
    read_errors = []
    for path in sorted(actual & expected):
        try:
            with h5py.File(path, "r") as handle:
                if set(handle.keys()) != {"flow"}:
                    raise ValueError(f"root keys={sorted(handle.keys())}")
                dataset = handle["flow"]
                shape = tuple(dataset.shape)
                dtype = str(dataset.dtype)
                shape_counts[str(shape)] = shape_counts.get(str(shape), 0) + 1
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                array = dataset[()]
                if shape != IMAGE_SIZE + (2,) or dtype != "float32":
                    raise ValueError(f"shape={shape} dtype={dtype}")
                nan_count += int(np.isnan(array).sum())
                inf_count += int(np.isinf(array).sum())
                extreme_count += int((np.abs(array) > FLOW_EXTREME_THRESHOLD).sum())
                if array.size:
                    max_abs = max(max_abs, float(np.abs(array).max()))
        except Exception as error:
            read_errors.append(f"{path}: {type(error).__name__}: {error}")
    result = {
        "stage": stage,
        "expected_predictions": len(expected),
        "prediction_files": len(actual),
        "missing": len(missing),
        "missing_examples": [str(path) for path in missing[:20]],
        "extra": len(extra),
        "extra_examples": [str(path) for path in extra[:20]],
        "shape_counts": shape_counts,
        "dtype_counts": dtype_counts,
        "nan": nan_count,
        "inf": inf_count,
        "extreme_threshold": FLOW_EXTREME_THRESHOLD,
        "extreme_values": extreme_count,
        "max_abs": max_abs,
        "read_errors": read_errors[:20],
        "read_error_count": len(read_errors),
    }
    result["passed"] = not (
        result["missing"]
        or result["extra"]
        or result["nan"]
        or result["inf"]
        or result["read_error_count"]
        or result["prediction_files"] != result["expected_predictions"]
    )
    return result


def validate_hdf5(
    path: Path,
    expected_predictions: int,
    expected_groups: list[str] | None = None,
) -> dict:
    result = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "root_keys": [],
        "datasets": 0,
        "groups": 0,
        "empty_groups": 0,
        "dataset_shapes": {},
        "dataset_dtypes": {},
        "data_rows": 0,
        "data_rows_by_name": {},
        "schema_ok": False,
        "nan": 0,
        "inf": 0,
        "read_errors": [],
    }
    if not path.exists():
        result["passed"] = False
        return result
    try:
        with h5py.File(path, "r") as handle:
            result["root_keys"] = sorted(handle.keys())
            def visit(name, item):
                if isinstance(item, h5py.Group):
                    result["groups"] += 1
                    if len(item) == 0:
                        result["empty_groups"] += 1
                    return
                if not isinstance(item, h5py.Dataset):
                    return
                result["datasets"] += 1
                shape = str(tuple(item.shape))
                dtype = str(item.dtype)
                result["dataset_shapes"][shape] = result["dataset_shapes"].get(shape, 0) + 1
                result["dataset_dtypes"][dtype] = result["dataset_dtypes"].get(dtype, 0) + 1
                if item.ndim == 2 and item.shape[1] == 2:
                    rows = int(item.shape[0])
                    result["data_rows_by_name"][name] = rows
                    if name == "flow":
                        result["data_rows"] = rows
                array = item[()]
                if np.issubdtype(array.dtype, np.number):
                    result["nan"] += int(np.isnan(array).sum())
                    result["inf"] += int(np.isinf(array).sum())
            handle.visititems(visit)
            if expected_groups is None:
                flow = handle.get("flow")
                result["schema_ok"] = bool(
                    result["root_keys"] == ["flow"]
                    and result["groups"] == 0
                    and result["datasets"] == 1
                    and isinstance(flow, h5py.Dataset)
                    and flow.ndim == 2
                    and flow.shape[1] == 2
                    and flow.dtype == np.dtype("float16")
                    and flow.compression == "gzip"
                    and flow.compression_opts == 9
                )
            else:
                expected = set(expected_groups)
                result["schema_ok"] = bool(
                    set(result["root_keys"]) == expected
                    and result["groups"] == len(expected)
                    and result["datasets"] == len(expected)
                    and result["empty_groups"] == 0
                    and all(
                        len(handle[group]) == 1
                        and isinstance(handle[group].get("flow"), h5py.Dataset)
                        and handle[group]["flow"].ndim == 2
                        and handle[group]["flow"].shape[1] == 2
                        and handle[group]["flow"].dtype == np.dtype("float16")
                        and handle[group]["flow"].compression == "gzip"
                        and handle[group]["flow"].compression_opts == 9
                        for group in expected
                    )
                )
    except Exception as error:
        result["read_errors"].append(f"{type(error).__name__}: {error}")
    result["expected_predictions"] = expected_predictions
    result["passed"] = bool(
        result["exists"]
        and result["size_bytes"] > 0
        and result["datasets"] > 0
        and result["schema_ok"]
        and result["empty_groups"] == 0
        and result["nan"] == 0
        and result["inf"] == 0
        and not result["read_errors"]
    )
    return result


def finalize(inventory: dict) -> dict:
    clean_jobs = jobs_for("clean", inventory)
    robust_jobs = jobs_for("robust", inventory)
    integrity = {
        "checkpoint": str(CHECKPOINT.resolve()),
        "clean": validate_predictions("clean", clean_jobs),
        "robust": validate_predictions("robust", robust_jobs),
        "corruptions_covered": inventory["robust_corruptions"],
    }
    integrity["flow_submission"] = validate_hdf5(
        PROBE / "flow_submission.hdf5", len(clean_jobs)
    )
    integrity["flow_robustness"] = validate_hdf5(
        PROBE / "flow_robustness.hdf5",
        len(robust_jobs),
        ["clean", *inventory["robust_corruptions"]],
    )
    integrity["passed"] = bool(
        integrity["clean"]["passed"]
        and integrity["robust"]["passed"]
        and integrity["flow_submission"]["passed"]
        and integrity["flow_robustness"]["passed"]
    )
    json_write(PROBE / "inference_integrity.json", integrity)
    clean = integrity["clean"]
    robust = integrity["robust"]
    submission = integrity["flow_submission"]
    robustness = integrity["flow_robustness"]
    status = "READY_FOR_SUBMISSION" if integrity["passed"] else "BLOCKED"
    lines = [
        "# Official transfer probe submission audit",
        "",
        "## Model",
        "",
        "CAR-GR-WAFT",
        "checkpoint step = 2000",
        "LR = 2.5e-5",
        "",
        "## HDF5",
        "",
        f"flow_submission.hdf5\nabsolute path: {submission['path']}\nsize: {submission['size_bytes']} bytes\nvalidation: {'PASS' if submission['passed'] else 'FAIL'}",
        "",
        f"flow_robustness.hdf5\nabsolute path: {robustness['path']}\nsize: {robustness['size_bytes']} bytes\nvalidation: {'PASS' if robustness['passed'] else 'FAIL'}",
        "",
        "## Coverage",
        "",
        f"- clean sequence count: {len(inventory['clean_scenes'])}",
        f"- clean prediction count: {clean['prediction_files']} / {clean['expected_predictions']}",
        f"- robust corruption count: {len(inventory['robust_corruptions'])}",
        f"- robust prediction count: {robust['prediction_files']} / {robust['expected_predictions']}",
        f"- missing = {clean['missing'] + robust['missing']}",
        f"- NaN = {clean['nan'] + robust['nan'] + submission['nan'] + robustness['nan']}",
        f"- Inf = {clean['inf'] + robust['inf'] + submission['inf'] + robustness['inf']}",
        "",
        "## Inference semantics",
        "",
        "Any inference-semantic changes: NO",
        "",
        f"Submission status: {status}",
        "",
    ]
    (PROBE / "submission_audit.md").write_text("\n".join(lines))
    return integrity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true", help="validate predictions and HDF5")
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--writers", type=int, default=4)
    parser.add_argument("--stage", choices=("both", "clean", "robust"), default="both")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.writers <= 0:
        parser.error("--batch-size and --writers must be positive")
    inventory = scan_data()
    if args.finalize:
        print(json.dumps(finalize(inventory), indent=2, sort_keys=True))
        return
    if not CHECKPOINT.exists():
        raise RuntimeError(f"missing winner checkpoint: {CHECKPOINT}")
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*.flo5")) and not args.resume:
        raise RuntimeError(
            f"refusing to overwrite existing predictions under {OUTPUT_ROOT}; use a new isolated output root"
        )
    write_frozen_manifest(inventory)
    model, head, device = load_frozen_model()
    if args.stage in ("both", "clean"):
        run_stage("clean", jobs_for("clean", inventory), model, head, device, args.batch_size, args.resume, args.writers)
    if args.stage in ("both", "robust"):
        run_stage("robust", jobs_for("robust", inventory), model, head, device, args.batch_size, args.resume, args.writers)
    if not args.no_finalize:
        print(json.dumps(finalize(inventory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
