"""P0 protocol lock and the canonical CAR-GR local evaluator."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/external_benchmark/sources/WAFT"
sys.path[:0] = [str(ROOT), str(SOURCE)]

from experiments import car_vs_cargr as C
from experiments.external_benchmark import local_protocol
from experiments.residual_screen import residual_experiment as R
import train_stage34 as S


OUT = ROOT / "experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1"
PARENT = ROOT / "results/car_vs_cargr/CAR_GR/best.pt"
HISTORICAL = ROOT / "results/car_vs_cargr/CAR_GR/evaluations/eval_step_2000.json"
U1 = ROOT / "experiments/CAR_GR_REWARD_GATE_U1/metrics/parent_clean.json"
CONFIG = ROOT / "experiments/external_benchmark/sources/WAFT/config/a2/dav2/tar-c-t-spring-540p.json"
NATIVE_SIZE = (540, 960)
FULL_SIZE = (1080, 1920)
EVAL_BATCH = 4
EVAL_SAMPLES = 64
EVAL_ITERS = 5
CONDITIONS = tuple(R.ALL_CONDITIONS)
THRESHOLD_PCT = 0.02


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def set_runtime() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def rows() -> list[dict]:
    selected = R._eval_rows(EVAL_SAMPLES)
    identities = [row["sample_id"] for row in selected]
    if identities != sorted(identities) or len(set(identities)) != EVAL_SAMPLES:
        raise RuntimeError("canonical dev64 identities are not unique and sorted")
    return selected


def load_parent(device: torch.device):
    args = S.load_config(CONFIG)
    args.algorithm = "waft-a2"
    args.image_size = list(NATIVE_SIZE)
    args.precision = "bf16"
    args.device = "cuda"
    args.iters = EVAL_ITERS
    model = S.build_model(args)
    payload = torch.load(PARENT, map_location="cpu", weights_only=True)
    identity = {name: payload.get(name) for name in ("contract_id", "branch", "step")}
    expected = {"contract_id": "car-vs-cargr-v1", "branch": "CAR_GR", "step": 2000}
    if identity != expected:
        raise RuntimeError(f"unexpected parent identity: {identity}")
    model.load_state_dict(payload["model"], strict=True)
    head = R.RiskGatedResidual(hidden=64)
    head.load_state_dict(payload["head"], strict=True)
    model.requires_grad_(False).eval().to(device)
    head.requires_grad_(False).eval().to(device)
    return args, model, head, identity


def runtime_contract(args, identity: dict) -> dict:
    device = torch.cuda.current_device()
    return {
        "contract_id": "car-gr-eval-protocol-lock-v1",
        "checkpoint": str(PARENT),
        "checkpoint_identity": identity,
        "checkpoint_load": "fresh torch.load(weights_only=True); model/head strict=True",
        "model_mode": "model.eval(); head.eval(); torch.no_grad()",
        "algorithm": args.algorithm,
        "evaluation_iterations": int(args.iters),
        "input_resolution": list(FULL_SIZE),
        "native_resolution": list(NATIVE_SIZE),
        "input_decode": "cv2 BGR -> copied RGB float32 [0,255]",
        "resize": "torch bilinear align_corners=True, 1080x1920 -> 540x960",
        "padding": "WAFT internal Padder factor=112, symmetric constant-zero; output unpadded",
        "model_precision": "BF16 autocast",
        "head_precision": "FP32 outside autocast",
        "autocast": "R._forward_bundle only; residual head excluded",
        "correlation_implementation": "WAFT native coords_grid + bilinear_sampler feature warp; no RAFT correlation volume",
        "dav2_feature_extraction": "Depth Anything V2 ViT-S, checkpoint-loaded, eval mode, frozen",
        "flow_upsampling": "WAFT learned convex 2x to 540x960, then bilinear align_corners=True to 1080x1920 with x/y scale=2",
        "valid_mask": "Spring GT valid mask AND finite GT/prediction",
        "max_flow_filtering": "none in metric aggregation",
        "scene_list": list(R._read_lines(R.DEV_SCENES)),
        "sample_selection": "R._eval_rows(64): sorted balanced scene x direction x side, four samples per stratum",
        "sides": ["left", "right"],
        "directions": ["BW", "FW"],
        "image_ordering": "sample_id ascending; image1/image2 follow labeled manifest direction",
        "conditions": list(CONDITIONS),
        "prediction_cache_generation": False,
        "prediction_cache_reading": False,
        "image_cache": "process-local decoded/corrupted image cache only; never shared between RUN_A/RUN_B",
        "aggregation": "global valid-pixel clean metrics; corruption proxy is macro mean of condition-global EPE",
        "per_image_weighting": "none",
        "per_pixel_weighting": "uniform over valid pixels",
        "data_loader": "none; direct manifest traversal",
        "data_order": "64 sorted identities, contiguous batches of 4",
        "batch_size": EVAL_BATCH,
        "seed": 0,
        "deterministic_flags": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "runtime": {
            "pid": os.getpid(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }


def summarize(records: list[dict], motion: dict[str, list[float]]) -> dict:
    per_condition = local_protocol.group_statistics(records, "condition")
    clean_records = [record for record in records if record["condition"] == "clean"]
    clean = per_condition["clean"]
    corrupt = [value["epe"] for name, value in per_condition.items() if name != "clean"]
    groups = {
        group.title(): float(np.mean([per_condition[name]["epe"] for name in names]))
        for group, names in R.GROUPS.items()
    }
    return {
        "clean": clean,
        "fixed_corruption_proxy_epe": float(np.mean(corrupt)),
        "per_condition": per_condition,
        "groups": groups,
        "per_scene_clean": local_protocol.group_statistics(clean_records, "scene"),
        "motion_epe_clean": {name: values[0] / values[1] for name, values in motion.items()},
        "motion_valid_count_clean": {name: int(values[1]) for name, values in motion.items()},
        "record_count": len(records),
    }


def evaluate(run_name: str) -> None:
    if run_name not in {"RUN_A", "RUN_B"}:
        raise ValueError("run_name must be RUN_A or RUN_B")
    run_dir = OUT / run_name
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite or retry {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "status.json", {"state": "RUNNING", "run": run_name})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("P0 canonical evaluation requires CUDA")
        torch.cuda.set_device(0)
        set_runtime()
        device = torch.device("cuda:0")
        args, model, head, identity = load_parent(device)
        torch.cuda.reset_peak_memory_stats()
        selected = rows()
        sample_manifest = [
            {name: str(row[name]) for name in ("sample_id", "scene", "direction", "side", "image1", "image2", "gt")}
            for row in selected
        ]
        records = []
        motion = {name: [0.0, 0] for name in ("s0-10", "s10-40", "s40+")}
        started = time.perf_counter()
        observed_bundle_dtypes = None
        with torch.no_grad():
            for start in range(0, len(selected), EVAL_BATCH):
                batch = selected[start : start + EVAL_BATCH]
                ground_truth = [R._read_eval_gt(row["gt"]) for row in batch]
                if any(item[0].shape[:2] != FULL_SIZE for item in ground_truth):
                    raise RuntimeError("canonical evaluator requires 1080x1920 Spring GT")
                for condition in CONDITIONS:
                    images = [R._condition_images(row, condition) for row in batch]
                    image1 = R._native_batch([pair[0] for pair in images], device, *NATIVE_SIZE)
                    image2 = R._native_batch([pair[1] for pair in images], device, *NATIVE_SIZE)
                    bundle = R._forward_bundle(model, image1, image2, device, torch.bfloat16)
                    output = head(bundle)
                    prediction = R._full_flow(bundle["f0"] + output["delta"], *FULL_SIZE, *NATIVE_SIZE)
                    if observed_bundle_dtypes is None:
                        observed_bundle_dtypes = {
                            "input": str(image1.dtype),
                            "f0": str(bundle["f0"].dtype),
                            "features": str(bundle["features"].dtype),
                            "head_parameter": str(next(head.parameters()).dtype),
                            "delta": str(output["delta"].dtype),
                        }
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError(f"non-finite prediction in {condition} batch {start}")
                    for index, row in enumerate(batch):
                        gt, valid = ground_truth[index]
                        statistics = local_protocol.accuracy_sufficient_statistics(prediction[index], gt, valid)
                        records.append({
                            "sample_id": str(row["sample_id"]),
                            "scene": str(row["scene"]),
                            "direction": str(row["direction"]),
                            "side": str(row["side"]),
                            "condition": condition,
                            "statistics": statistics,
                        })
                        if condition == "clean":
                            error = np.linalg.norm(prediction[index] - gt, axis=-1)
                            magnitude = np.linalg.norm(gt, axis=-1)
                            base = valid & np.isfinite(error) & np.isfinite(magnitude)
                            selectors = {
                                "s0-10": magnitude < 10,
                                "s10-40": (magnitude >= 10) & (magnitude < 40),
                                "s40+": magnitude >= 40,
                            }
                            for name, selector in selectors.items():
                                mask = base & selector
                                motion[name][0] += float(error[mask].sum(dtype=np.float64))
                                motion[name][1] += int(mask.sum())
                    del images, image1, image2, bundle, output, prediction
                print(json.dumps({"run": run_name, "samples": start + len(batch), "total": EVAL_SAMPLES}), flush=True)
        result = {
            "run": run_name,
            "runtime_contract": runtime_contract(args, identity),
            "observed_dtypes": observed_bundle_dtypes,
            "sample_manifest": sample_manifest,
            "summary": summarize(records, motion),
            "records": records,
            "wall_clock_s": time.perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        }
        atomic_json(run_dir / "result.json", result)
        atomic_json(run_dir / "status.json", {"state": "COMPLETED", "run": run_name})
        print(json.dumps({"run": run_name, "clean_epe": result["summary"]["clean"]["epe"]}), flush=True)
    except BaseException:
        error = traceback.format_exc()
        atomic_text(run_dir / "exception.txt", error)
        atomic_json(run_dir / "status.json", {"state": "BLOCKED", "run": run_name, "exception": error})
        atomic_json(OUT / "status.json", {"state": "P0_BLOCKED_RUNTIME_ERROR", "run": run_name, "exception": error})
        raise


def audit() -> None:
    if (OUT / "protocol_audit.json").exists():
        raise RuntimeError("refusing to overwrite the P0 audit")
    historical = json.loads(HISTORICAL.read_text())
    u1 = json.loads(U1.read_text())
    selected = rows()
    strata = {}
    for row in selected:
        key = f"{row['scene']}:{row['direction']}:{row['side']}"
        strata[key] = strata.get(key, 0) + 1
    fields = {
        "checkpoint_path": [str(PARENT), str(PARENT)],
        "checkpoint_state": ["in-process step-2000 model/head, then saved as best.pt", "fresh best.pt reload"],
        "model_mode": ["model.eval(); head.eval()", "model.eval(); head.eval()"],
        "strict_load": ["not reloaded at evaluation; initial parent load was strict", "fresh model/head strict=True"],
        "evaluation_iterations": [5, 5],
        "input_resolution": [list(FULL_SIZE), list(FULL_SIZE)],
        "native_resolution": [list(NATIVE_SIZE), list(NATIVE_SIZE)],
        "resize": ["bilinear align_corners=True", "bilinear align_corners=True"],
        "padding": ["WAFT Padder factor=112; zero pad and unpad", "same"],
        "precision": ["WAFT BF16, residual head FP32", "WAFT and residual head BF16"],
        "autocast": ["R._forward_bundle only", "model + bundle + residual head in one BF16 scope"],
        "correlation_implementation": ["WAFT coords_grid + bilinear_sampler feature warp", "same"],
        "dav2_feature_extraction": ["DAv2 ViT-S eval-mode forward", "same, inside wider BF16 scope"],
        "flow_upsampling": ["WAFT learned convex 2x then bilinear x2 with vector scaling", "same"],
        "valid_mask": ["GT valid AND finite GT/prediction", "same"],
        "max_flow_filtering": ["none in metrics", "none in metrics; 40000 only diagnostic"],
        "scene_list": [list(R._read_lines(R.DEV_SCENES)), list(R._read_lines(R.DEV_SCENES))],
        "sample_list": ["same R._eval_rows(64)", "same R._eval_rows(64)"],
        "left_right": ["balanced left/right", "balanced left/right"],
        "fw_bw": ["balanced FW/BW", "balanced FW/BW"],
        "image_ordering": ["sample_id sorted", "sample_id sorted"],
        "cache_generation": ["no prediction cache; process-local image cache", "CPU parent-environment cache built after live parent prediction"],
        "cache_reading_for_parent_metric": [False, False],
        "aggregation": ["global valid-pixel", "global valid-pixel"],
        "per_image_weighting": ["none", "none"],
        "per_pixel_weighting": ["uniform valid pixels", "uniform valid pixels"],
        "data_loader_order": ["direct manifest, batch 8", "direct manifest, batch 8"],
        "deterministic_flags": ["seed 0; cudnn benchmark=True; deterministic=False", "same"],
    }
    report = {
        "historical_car_gr_epe": float(historical["clean_epe"]),
        "u1_fresh_reload_epe": float(u1["epe"]),
        "relative_difference_pct": 100.0 * (float(u1["epe"]) / float(historical["clean_epe"]) - 1.0),
        "actual_runtime_comparison": {
            name: {"historical_car_gr": values[0], "u1_fresh_reload": values[1]}
            for name, values in fields.items()
        },
        "material_protocol_difference": "residual-head autocast scope (FP32 historical/official-transfer versus BF16 U1)",
        "attribution_boundary": "The semantic mismatch makes the absolute EPE values non-comparable; its exact numeric share is not claimed without a one-factor experiment.",
        "canonical_selection": "fresh strict-load official-transfer semantics: WAFT BF16, expanded bundle and RiskGatedResidual FP32, batch 4",
        "canonical_selection_evidence": [
            "tools/official_transfer_probe.py: load_frozen_model/run_stage",
            "experiments/residual_screen/residual_experiment.py: _forward_bundle/_native_batch/_full_flow",
            "results/car_vs_cargr/official_transfer_probe/clean_inference.log",
        ],
        "official_transfer_runtime_note": "The existing transfer logs record historical resume segments with batch 4 and 16; P0 locks the source default batch 4 and never mixes batches.",
        "sample_ids": [str(row["sample_id"]) for row in selected],
        "strata_counts": strata,
    }
    atomic_json(OUT / "protocol_audit.json", report)
    lines = [
        "# P0 evaluation protocol audit",
        "",
        f"Historical CAR-GR clean EPE: `{report['historical_car_gr_epe']:.9f}`.",
        f"U1 fresh-reload clean EPE: `{report['u1_fresh_reload_epe']:.9f}` (`{report['relative_difference_pct']:+.6f}%`).",
        "",
        "The material runtime mismatch is the residual-head autocast scope: historical CAR-GR and official-transfer run WAFT under BF16 autocast but execute the expanded-bundle RiskGatedResidual in FP32; U1 executes both inside BF16 autocast. The exact numerical share is not claimed because this P0 is a lock, not an additional factor-isolation experiment.",
        "",
        "The canonical evaluator is therefore a fresh strict-load, eval-mode, no-cache implementation of official-transfer semantics at 540x960, 5 iterations, batch 4. Full resolved values and all 64 sample identities are in `protocol_audit.json`; RUN_A/RUN_B record observed runtime values independently.",
    ]
    atomic_text(OUT / "feature_scope_note.md", "\n".join(lines) + "\n")
    atomic_json(OUT / "status.json", {"state": "P0_AUDIT_COMPLETE"})


def compare_and_lock() -> None:
    a = json.loads((OUT / "RUN_A/result.json").read_text())
    b = json.loads((OUT / "RUN_B/result.json").read_text())
    if a["sample_manifest"] != b["sample_manifest"]:
        raise RuntimeError("RUN_A/RUN_B sample identity mismatch")
    a_records = {(row["sample_id"], row["condition"]): row for row in a["records"]}
    b_records = {(row["sample_id"], row["condition"]): row for row in b["records"]}
    if a_records.keys() != b_records.keys():
        raise RuntimeError("RUN_A/RUN_B per-sample record mismatch")
    comparisons = []
    for key in sorted(a_records):
        left, right = a_records[key]["statistics"], b_records[key]["statistics"]
        if left["valid_count"] != right["valid_count"]:
            raise RuntimeError(f"valid-pixel mismatch for {key}")
        left_epe = left["epe_sum"] / left["valid_count"]
        right_epe = right["epe_sum"] / right["valid_count"]
        comparisons.append({
            "sample_id": key[0],
            "condition": key[1],
            "run_a_epe": left_epe,
            "run_b_epe": right_epe,
            "absolute_difference": abs(right_epe - left_epe),
            "relative_difference_pct": 100.0 * abs(right_epe - left_epe) / left_epe,
            "valid_count": int(left["valid_count"]),
        })
    a_epe = float(a["summary"]["clean"]["epe"])
    b_epe = float(b["summary"]["clean"]["epe"])
    relative = 100.0 * abs(b_epe - a_epe) / a_epe
    scene_comparisons = []
    for scene in sorted(a["summary"]["per_scene_clean"]):
        left = float(a["summary"]["per_scene_clean"][scene]["epe"])
        right = float(b["summary"]["per_scene_clean"][scene]["epe"])
        scene_comparisons.append({
            "scene": scene,
            "run_a_epe": left,
            "run_b_epe": right,
            "absolute_difference": abs(right - left),
            "relative_difference_pct": 100.0 * abs(right - left) / left,
        })
    passed = relative <= THRESHOLD_PCT
    comparison = {
        "run_a_clean_epe": a_epe,
        "run_b_clean_epe": b_epe,
        "relative_epe_difference_pct": relative,
        "threshold_pct": THRESHOLD_PCT,
        "samples_identical": True,
        "valid_pixels_identical": True,
        "per_sample_identity_identical": True,
        "max_per_sample_relative_difference_pct": max(row["relative_difference_pct"] for row in comparisons),
        "scene_comparisons": scene_comparisons,
        "per_sample_condition_comparisons": comparisons,
        "passed": passed,
    }
    atomic_json(OUT / "run_comparison.json", comparison)
    if not passed:
        atomic_text(OUT / "decision.md", "EVAL_PROTOCOL_NOT_STABLE\n")
        atomic_json(OUT / "status.json", {"state": "P0_BLOCKED_PROTOCOL_UNSTABLE", **comparison})
        print("EVAL_PROTOCOL_NOT_STABLE", flush=True)
        return
    contract = dict(a["runtime_contract"])
    contract.update({
        "locked_from": ["RUN_A", "RUN_B"],
        "stability_threshold_pct": THRESHOLD_PCT,
        "observed_relative_epe_difference_pct": relative,
        "canonical_reference_run": "RUN_A",
        "canonical_parent_metrics": a["summary"],
        "required_for_phases": ["P1", "P2", "P3", "P4", "P5"],
    })
    contract["runtime"].pop("pid", None)
    atomic_json(OUT / "canonical_eval_contract.json", contract)
    atomic_text(OUT / "decision.md", "P0_PASS_PROTOCOL_LOCK\n")
    atomic_json(OUT / "status.json", {
        "state": "P0_PASS_PROTOCOL_LOCK",
        "run_a_clean_epe": a_epe,
        "run_b_clean_epe": b_epe,
        "relative_epe_difference_pct": relative,
    })
    print("P0_PASS_PROTOCOL_LOCK", flush=True)


def self_check() -> None:
    assert EVAL_SAMPLES == 64 and EVAL_BATCH == 4 and EVAL_ITERS == 5
    assert CONDITIONS[0] == "clean" and len(CONDITIONS) == 19
    assert len(rows()) == EVAL_SAMPLES
    print("P0 canonical evaluator self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "audit", "evaluate", "lock"))
    parser.add_argument("--run", choices=("RUN_A", "RUN_B"))
    args = parser.parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "audit":
        audit()
    elif args.command == "evaluate":
        if not args.run:
            parser.error("evaluate requires --run")
        evaluate(args.run)
    else:
        compare_and_lock()


if __name__ == "__main__":
    main()
