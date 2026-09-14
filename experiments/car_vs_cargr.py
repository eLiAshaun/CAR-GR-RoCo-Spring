"""Serial CAR-WAFT versus CAR-GR-WAFT comparison on the fixed local proxy."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/external_benchmark/sources/WAFT"
sys.path[:0] = [str(ROOT), str(SOURCE)]

from experiments.residual_screen import residual_experiment as R
from criterion.loss import sequence_loss
from dataloader.paired_corruptions import PairedSpringCorruptionDataset
import train_stage34 as S


CONTRACT_ID = "car-vs-cargr-v1"
BRANCHES = ("CAR", "CAR_GR")
EVAL_POINTS = (0, 500, 1000, 2000, 3000, 4500, 6000)
EFFECTIVE_BATCH = 8
SPRING_FINETUNE_LR = 1e-4
ROBUST_FINETUNE_LR = 0.25 * SPRING_FINETUNE_LR


class DrawDataset(PairedSpringCorruptionDataset):
    """Allow a deterministic epoch token so A and B see the same draws."""

    def __getitem__(self, token):
        index, epoch = token
        self.epoch = int(epoch)
        return super().__getitem__(int(index))


class ZeroResidual(nn.Module):
    def forward(self, bundle):
        return {"delta": torch.zeros_like(bundle["f0"])}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_args(height: int, width: int):
    args = S.load_config(R.CONFIG)
    args.algorithm = "waft-a2"
    args.image_size = [height, width]
    args.precision = "bf16"
    args.lr = ROBUST_FINETUNE_LR
    args.device = "cuda"
    return args


def build_models(args, branch: str):
    student = S.build_model(args)
    S.load_parent_checkpoint(student, R.PARENT_CHECKPOINT, allow_new=False)
    for name, parameter in student.named_parameters():
        parameter.requires_grad = not name.startswith("encoder.encoder.")
    teacher = S.build_model(args)
    S.load_parent_checkpoint(teacher, R.PARENT_CHECKPOINT, allow_new=False)
    teacher.requires_grad_(False).eval()
    head = R.RiskGatedResidual(hidden=64) if branch == "CAR_GR" else ZeroResidual()
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable += list(head.parameters())
    audit = {
        "backbone_frozen": all(not parameter.requires_grad for name, parameter in student.named_parameters() if name.startswith("encoder.encoder.")),
        "student_trainable_parameters": sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad),
        "residual_trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
    }
    return student, teacher, head, trainable, audit


def draw_tokens(size: int, count: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    result = []
    epoch = 0
    while len(result) < count:
        result.extend((int(index), epoch) for index in rng.permutation(size))
        epoch += 1
    return result[:count]


def data_loader(args, micro_batch: int, start_step: int, steps: int, workers: int):
    dataset = DrawDataset(
        root=ROOT / "data/spring",
        split="train",
        crop_size=(int(args.image_size[0]), int(args.image_size[1])),
        min_scale=-1.0,
        max_scale=-0.8,
        do_flip=True,
        paired=True,
        scene_names=R._read_lines(R.TRAIN_SCENES),
        corruption_names=R.TRAIN_CORRUPTIONS,
        severity_min=1,
        severity_max=4,
        seed=0,
        balanced_groups=True,
    )
    if len(dataset) != 14_688:
        raise RuntimeError(f"expected 14,688 Spring training pairs, found {len(dataset):,}")
    tokens = draw_tokens(len(dataset), steps * EFFECTIVE_BATCH, seed=0)[start_step * EFFECTIVE_BATCH:]
    options = {
        "batch_size": micro_batch,
        "sampler": tokens,
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if workers:
        options.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **options)


def bundle_from_output(output):
    aux = output["aux"]
    fmap1, fmap2 = aux["fmap1"].float(), aux["fmap2"].float()
    flow_low = aux["flow_low"].float()
    coords = R.WAFT_UTILS.coords_grid(flow_low.shape[0], flow_low.shape[-2], flow_low.shape[-1], flow_low.device) + flow_low
    warped = R.WAFT_UTILS.bilinear_sampler(fmap2, coords.permute(0, 2, 3, 1))
    q = (F.normalize(fmap1, dim=1, eps=1e-6) * F.normalize(warped, dim=1, eps=1e-6)).sum(1, keepdim=True)
    return R._expand_native_bundle({
        "f0": output["flow"][-1].float(),
        "hidden_low": aux["hidden"].float(),
        "q_low": q,
        "last_update_low": aux["last_update"].float(),
    })


def corrected_output(output, prediction, flow_gt):
    corrected = dict(output)
    corrected["flow"] = list(output["flow"])
    corrected["flow"][-1] = prediction
    info = output["info"][-1]
    raw_b, weight = info[:, 2:], info[:, :2]
    log_b = torch.zeros_like(raw_b)
    log_b[:, 0] = raw_b[:, 0].clamp(0, 10)
    log_b[:, 1] = raw_b[:, 1].clamp(-0.0, 0)
    term2 = (flow_gt - prediction).abs().unsqueeze(2) * torch.exp(-log_b).unsqueeze(1)
    term1 = weight - math.log(2) - log_b
    corrected["nf"] = list(output["nf"])
    corrected["nf"][-1] = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
    return corrected


def valid_mask(flow, valid):
    return (valid >= 0.5) & torch.isfinite(flow).all(1) & (torch.linalg.vector_norm(flow, dim=1) < 40_000)


def anchor_loss(prediction, target, flow, valid):
    mask = valid_mask(flow, valid) & torch.isfinite(prediction).all(1) & torch.isfinite(target).all(1)
    return torch.sqrt((prediction.float() - target.float()).square().sum(1) + 1e-6)[mask].mean()


def gate_terms(output, bundle, flow, valid):
    mask = valid_mask(flow, valid)
    gate = output["gate"].float()[:, 0]
    target = torch.sigmoid((torch.linalg.vector_norm(bundle["f0"].detach() - flow, dim=1) - 0.5) / 0.15)
    selected_gate = gate.clamp(1e-5, 1 - 1e-5)[mask]
    selected_target = target.float()[mask]
    gate_loss = -(selected_target * selected_gate.log() + (1 - selected_target) * (1 - selected_gate).log()).mean()
    correction_reg = output["delta"].float().abs().sum(1)[mask].mean()
    return gate_loss, correction_reg


def train_microbatch(student, teacher, head, branch, batch, device, scale):
    clean1 = batch["clean_image1"].to(device, non_blocking=True)
    clean2 = batch["clean_image2"].to(device, non_blocking=True)
    corrupt1 = batch["corrupt_image1"].to(device, non_blocking=True)
    corrupt2 = batch["corrupt_image2"].to(device, non_blocking=True)
    flow = batch["flow"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    with torch.no_grad(), S._autocast(device, "bf16"):
        teacher_clean = teacher(clean1, clean2)["flow"][-1].detach()
    with S._autocast(device, "bf16"):
        clean_output = student(clean1, clean2, flow_gt=flow, return_aux=branch == "CAR_GR")
        clean_prediction = clean_output["flow"][-1]
        clean_gate = clean_reg = clean_prediction.new_zeros(())
        if branch == "CAR_GR":
            clean_bundle = bundle_from_output(clean_output)
            clean_head = head(clean_bundle)
            clean_prediction = clean_bundle["f0"] + clean_head["delta"]
            clean_output = corrected_output(clean_output, clean_prediction, flow)
            clean_gate, clean_reg = gate_terms(clean_head, clean_bundle, flow, valid)
        clean_gt = sequence_loss(clean_output, flow, valid, gamma=0.85)
        anchor = anchor_loss(clean_prediction, teacher_clean, flow, valid)
        clean_objective = 0.5 * clean_gt + 0.1 * anchor + 0.025 * clean_gate + 0.005 * clean_reg
    (scale * clean_objective).backward()
    del clean_output, clean_prediction, teacher_clean
    with S._autocast(device, "bf16"):
        corrupt_output = student(corrupt1, corrupt2, flow_gt=flow, return_aux=branch == "CAR_GR")
        corrupt_gate = corrupt_reg = corrupt_output["flow"][-1].new_zeros(())
        if branch == "CAR_GR":
            corrupt_bundle = bundle_from_output(corrupt_output)
            corrupt_head = head(corrupt_bundle)
            corrupt_prediction = corrupt_bundle["f0"] + corrupt_head["delta"]
            corrupt_output = corrected_output(corrupt_output, corrupt_prediction, flow)
            corrupt_gate, corrupt_reg = gate_terms(corrupt_head, corrupt_bundle, flow, valid)
        corrupt_gt = sequence_loss(corrupt_output, flow, valid, gamma=0.85)
        corrupt_objective = 0.5 * corrupt_gt + 0.025 * corrupt_gate + 0.005 * corrupt_reg
    (scale * corrupt_objective).backward()
    total = clean_objective.detach() + corrupt_objective.detach()
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite CAR objective: {float(total)}")
    return {
        "loss": float(total),
        "clean_gt": float(clean_gt.detach()),
        "corrupt_gt": float(corrupt_gt.detach()),
        "anchor": float(anchor.detach()),
        "gate": float((0.5 * (clean_gate + corrupt_gate)).detach()),
        "correction_reg": float((0.5 * (clean_reg + corrupt_reg)).detach()),
    }


def compact(report):
    return {
        "step": report["step"],
        "clean_epe": report["clean_epe"],
        "corrupt_epe": report["corrupt_epe"],
        "proxy_rbs": report["proxy_rbs"],
        "groups": report["groups"],
        "one_px": report["one_px"],
        "three_px": report["three_px"],
        "motion_epe": report["motion_epe"],
        **({"gate_analysis": report["gate_analysis"]} if "gate_analysis" in report else {}),
    }


def normalize_report(report, baseline):
    report["correction_proxy_rbs_vs_pre_correction"] = report.pop("proxy_rbs_vs_baseline")
    report["correction_delta_clean_pct"] = report.pop("delta_clean_pct")
    report["correction_delta_corrupt_pct"] = report.pop("delta_corrupt_pct")
    report["proxy_rbs"] = 0.5 * (report["clean_epe"] / baseline["clean_epe"] + report["corrupt_epe"] / baseline["corrupt_epe"])
    report["delta_clean_pct"] = 100 * (report["clean_epe"] / baseline["clean_epe"] - 1)
    report["delta_corrupt_pct"] = 100 * (report["corrupt_epe"] / baseline["corrupt_epe"] - 1)
    report["groups"] = {
        group.title(): float(np.mean([report["per_condition"][name]["epe"] for name in names]))
        for group, names in R.GROUPS.items()
    }
    report["contract_id"] = CONTRACT_ID
    report["metric_scope"] = f"{report['eval_sample_count']}-pair Spring dev synthetic-corruption local proxy; not official RobustSpring scoring"
    return report


def evaluate(student, head, branch, step, args, output_root: Path, baseline):
    student.eval()
    head.eval()
    output_dir = output_root / branch / "evaluations"
    report = R._evaluate_head(
        student, head, branch, R._eval_rows(int(args.eval_samples)), torch.device("cuda"), torch.bfloat16,
        int(args.height), int(args.width), int(args.eval_batch), output_dir, step, "bf16", None,
    )
    if baseline is None:
        baseline = {"clean_epe": report["clean_epe"], "corrupt_epe": report["corrupt_epe"]}
    normalize_report(report, baseline)
    atomic_json(output_dir / f"eval_step_{step:04d}.json", report)
    return report


def checkpoint_payload(branch, step, student, head, optimizer=None, scheduler=None):
    payload = {"contract_id": CONTRACT_ID, "branch": branch, "step": step, "model": student.state_dict(), "head": head.state_dict()}
    if optimizer is not None:
        payload.update(
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        )
    return payload


def save_training_state(run_dir, branch, step, student, head, optimizer, scheduler, best=False):
    R._atomic_torch(run_dir / ("best.pt" if best else "latest.pt"), checkpoint_payload(
        branch, step, student, head, None if best else optimizer, None if best else scheduler,
    ))


def run_branch(branch: str, args, output_root: Path):
    run_dir = output_root / branch
    status_path = run_dir / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("complete"):
        return json.loads((run_dir / "best.json").read_text())
    seed_everything(0)
    config = model_args(args.height, args.width)
    student, teacher, head, trainable, audit = build_models(config, branch)
    device = torch.device("cuda")
    student.to(device)
    teacher.to(device)
    head.to(device)
    optimizer = torch.optim.AdamW(trainable, lr=ROBUST_FINETUNE_LR, weight_decay=1e-5, eps=1e-8)
    scheduler = S._scheduler(optimizer, int(args.steps), 0.05)
    micro_batch = int(args.car_batch if branch == "CAR" else args.cargr_batch)
    if EFFECTIVE_BATCH % micro_batch:
        raise ValueError("micro batch must divide the effective batch of 8")
    accumulation = EFFECTIVE_BATCH // micro_batch
    start_step = 0
    curve = []
    latest = run_dir / "latest.pt"
    if latest.exists():
        payload = torch.load(latest, map_location=device, weights_only=True)
        if payload.get("contract_id") != CONTRACT_ID or payload.get("branch") != branch:
            raise RuntimeError(f"incompatible checkpoint: {latest}")
        student.load_state_dict(payload["model"], strict=True)
        head.load_state_dict(payload["head"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        start_step = int(payload["step"])
        curve = json.loads((run_dir / "curve.json").read_text())
    baseline_path = output_root / "baseline.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if not curve:
        report = evaluate(student, head, branch, 0, args, output_root, baseline)
        if baseline is None:
            baseline = compact(report)
            baseline["contract_id"] = CONTRACT_ID
            baseline["metric_scope"] = report["metric_scope"]
            baseline["training_contract"] = {
                "parent": "WAFT-DAv2-A2",
                "train_pairs": 14_688,
                "train_scene_file": str(R.TRAIN_SCENES.relative_to(ROOT)),
                "corruptions": list(R.TRAIN_CORRUPTIONS),
                "corruption_family_weights": {name.title(): 0.2 for name in R.GROUPS},
                "seed": 0,
                "crop": [int(args.height), int(args.width)],
                "effective_batch": EFFECTIVE_BATCH,
                "optimizer": "AdamW",
                "validated_spring_finetune_lr": SPRING_FINETUNE_LR,
                "robust_finetune_lr": ROBUST_FINETUNE_LR,
                "schedule": "cosine with 5% warmup",
                "max_steps": int(args.steps),
                "evaluation_steps": [step for step in EVAL_POINTS if step <= int(args.steps)],
                "precision": "BF16 + TF32",
                "teacher": "frozen original WAFT",
                "backbone": "DAv2 ViT frozen; WAFT DPT/task/updater/flow parameters trainable",
            }
            atomic_json(baseline_path, baseline)
        curve = [compact(report)]
        atomic_json(run_dir / "curve.json", curve)
        atomic_json(run_dir / "best.json", curve[0])
        if branch == "CAR_GR":
            atomic_json(run_dir / "gate_analysis.json", report["gate_analysis"])
        save_training_state(run_dir, branch, 0, student, head, optimizer, scheduler)
        save_training_state(run_dir, branch, 0, student, head, optimizer, scheduler, best=True)
    best = min(curve, key=lambda row: row["proxy_rbs"])
    best_corrupt = min(row["corrupt_epe"] for row in curve)
    no_improve = 0
    for row in reversed(curve[1:]):
        if row["proxy_rbs"] > best["proxy_rbs"] and row["corrupt_epe"] > best_corrupt:
            no_improve += 1
        else:
            break
    loader = data_loader(config, micro_batch, start_step, int(args.steps), int(args.workers))
    iterator = iter(loader)
    S._set_train_mode(student, config)
    head.train()
    optimizer.zero_grad(set_to_none=True)
    log_path = run_dir / "train_log.jsonl"
    started = time.perf_counter()
    early_stopped = False
    for step in range(start_step + 1, int(args.steps) + 1):
        totals = defaultdict(float)
        for _ in range(accumulation):
            metrics = train_microbatch(student, teacher, head, branch, next(iterator), device, 1 / accumulation)
            for name, value in metrics.items():
                totals[name] += value / accumulation
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise FloatingPointError("non-finite CAR gradient")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 25 == 0:
            record = {"step": step, "elapsed_s": time.perf_counter() - started, "lr": optimizer.param_groups[0]["lr"], **totals}
            run_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"branch": branch, **record}, sort_keys=True), flush=True)
        if step not in EVAL_POINTS and step != int(args.steps):
            continue
        report = evaluate(student, head, branch, step, args, output_root, baseline)
        row = compact(report)
        previous_proxy, previous_corrupt = best["proxy_rbs"], best_corrupt
        curve.append(row)
        atomic_json(run_dir / "curve.json", curve)
        save_training_state(run_dir, branch, step, student, head, optimizer, scheduler)
        if row["proxy_rbs"] < previous_proxy:
            best = row
            atomic_json(run_dir / "best.json", best)
            save_training_state(run_dir, branch, step, student, head, optimizer, scheduler, best=True)
            if branch == "CAR_GR":
                atomic_json(run_dir / "gate_analysis.json", report["gate_analysis"])
        best_corrupt = min(best_corrupt, row["corrupt_epe"])
        no_improve = no_improve + 1 if row["proxy_rbs"] >= previous_proxy and row["corrupt_epe"] >= previous_corrupt else 0
        atomic_json(status_path, {"contract_id": CONTRACT_ID, "branch": branch, "step": step, "complete": False, "early_stopped": False, "micro_batch": micro_batch, "effective_batch": EFFECTIVE_BATCH, "audit": audit})
        if no_improve >= 3:
            early_stopped = True
            break
        S._set_train_mode(student, config)
        head.train()
    atomic_json(status_path, {"contract_id": CONTRACT_ID, "branch": branch, "step": curve[-1]["step"], "complete": True, "early_stopped": early_stopped, "micro_batch": micro_batch, "effective_batch": EFFECTIVE_BATCH, "best_step": best["step"], "audit": audit})
    del loader, iterator, optimizer, scheduler, student, teacher, head
    gc.collect()
    torch.cuda.empty_cache()
    return best


def classify(baseline, car, cargr):
    a_better = 100 * (cargr["proxy_rbs"] / car["proxy_rbs"] - 1)
    b_better = 100 * (car["proxy_rbs"] / cargr["proxy_rbs"] - 1)
    b_clean_regression = 100 * (cargr["clean_epe"] / baseline["clean_epe"] - 1)
    if a_better >= 0.15 or b_clean_regression > 0.3:
        winner = "CAR_WAFT"
        selected = car
    elif b_better >= 0.15 and cargr["corrupt_epe"] <= car["corrupt_epe"] and b_clean_regression <= 0.3:
        winner = "CAR_GR_WAFT"
        selected = cargr
    else:
        winner = "STATISTICAL_TIE"
        selected = car
    improvement = 100 * (1 - selected["proxy_rbs"])
    clean_improvement = 100 * (1 - selected["clean_epe"] / baseline["clean_epe"])
    corrupt_improvement = 100 * (1 - selected["corrupt_epe"] / baseline["corrupt_epe"])
    ready = improvement >= 0.3 and ((corrupt_improvement >= 0.5 and clean_improvement >= 0) or (clean_improvement > 0 and corrupt_improvement > 0))
    return {
        "contract_id": CONTRACT_ID,
        "winner": winner,
        "decision": "SCALE_WINNER" if ready else "NEITHER_READY_FOR_SCALE",
        "relative_gap_pct": b_better if winner == "CAR_GR_WAFT" else a_better,
        "winner_vs_b0": {"proxy_rbs_improvement_pct": improvement, "clean_improvement_pct": clean_improvement, "corrupt_improvement_pct": corrupt_improvement},
        "baseline": baseline,
        "CAR": car,
        "CAR_GR": cargr,
        "metric_scope": "local proxy only; not official RobustSpring validation",
    }


def write_comparison(output_root: Path):
    baseline = json.loads((output_root / "baseline.json").read_text())
    car = json.loads((output_root / "CAR/best.json").read_text())
    cargr = json.loads((output_root / "CAR_GR/best.json").read_text())
    result = classify(baseline, car, cargr)
    atomic_json(output_root / "comparison.json", result)
    (output_root / "final_report.md").write_text(
        "# CAR-WAFT vs CAR-GR-WAFT\n\n"
        f"Winner: **{result['winner']}**  \nDecision: **{result['decision']}**\n\n"
        "| branch | step | clean EPE | corrupt EPE | ProxyRbS |\n|---|---:|---:|---:|---:|\n"
        f"| B0 | 0 | {baseline['clean_epe']:.9f} | {baseline['corrupt_epe']:.9f} | 1.000000000 |\n"
        f"| CAR | {car['step']} | {car['clean_epe']:.9f} | {car['corrupt_epe']:.9f} | {car['proxy_rbs']:.9f} |\n"
        f"| CAR-GR | {cargr['step']} | {cargr['clean_epe']:.9f} | {cargr['corrupt_epe']:.9f} | {cargr['proxy_rbs']:.9f} |\n\n"
        "These are fixed local Spring-dev synthetic-corruption proxy metrics, not official RobustSpring validation.\n"
    )
    return result


def self_check():
    head = R.RiskGatedResidual(hidden=64)
    bundle = {"features": torch.randn(1, 67, 8, 8), "gate_features": torch.randn(1, 67, 8, 8), "f0": torch.randn(1, 2, 8, 8)}
    output = head(bundle)
    assert torch.equal(output["delta"], torch.zeros_like(output["delta"]))
    assert torch.equal(output["gate"], torch.full_like(output["gate"], 0.5))
    baseline = {"clean_epe": 1.0, "corrupt_epe": 2.0}
    a = {"clean_epe": 0.999, "corrupt_epe": 1.99, "proxy_rbs": 0.997}
    b = {"clean_epe": 1.0, "corrupt_epe": 1.98, "proxy_rbs": 0.995}
    assert classify(baseline, a, b)["winner"] == "CAR_GR_WAFT"
    print("CAR/GR self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/car_vs_cargr")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--car-batch", type=int, default=8)
    parser.add_argument("--cargr-batch", type=int, default=8)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CAR comparison requires CUDA")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.run_all:
        for branch in BRANCHES:
            run_branch(branch, args, args.output_root)
        print(json.dumps(write_comparison(args.output_root), indent=2), flush=True)
    elif args.branch:
        print(json.dumps(run_branch(args.branch, args, args.output_root), indent=2), flush=True)
    elif args.compare:
        print(json.dumps(write_comparison(args.output_root), indent=2), flush=True)
    else:
        parser.error("choose --run-all, --branch, --compare, or --self-check")


if __name__ == "__main__":
    main()
