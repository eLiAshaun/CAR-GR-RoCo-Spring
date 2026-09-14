# CAR-GR-WAFT step-2000 reproducibility bundle

This is the frozen model that generated `results/car_vs_cargr/official_transfer_probe/flow_submission.hdf5`.

- model: CAR-GR-WAFT
- contract: `car-vs-cargr-v1`
- selected checkpoint: step 2000, chosen by the lowest local ProxyRbS
- checkpoint SHA256: `2321390c920727f23c7ad28479b1aea63442e58dbb45a00cfea82c0247b7221f`
- clean submission SHA256: `3e0937a45e05acd99ca9e8ef2ae88fb811384d4b5a0e31f2f47758b9b34a5b61`
- WAFT source base commit: `b152ff1cad1af8c185ee7b141997c48ff3334c87`

The source tree includes the exact local WAFT changes used by the checkpoint: auxiliary outputs in `model/waft_a2.py`, the CAR-GR residual path, paired corruptions, and the Stage 3/4 training helpers. The source was dirty relative to the commit above, so the included files, not the bare commit, are authoritative.

## What is included

- exact CAR-GR training and evaluator code
- portable checkpoint-only canonical evaluator
- original WAFT model source and modified local files
- WAFT parent checkpoint and CAR-GR step-2000 checkpoint
- training curve, logs, selected metrics, comparison, and provenance records
- clean `flow_submission.hdf5`
- official clean/robust subsampling executables
- pinned `environment.yml`
- SHA256 manifest generated as `BUNDLE_MANIFEST.sha256`

The 251 GB Spring dataset, RobustSpring images, raw `.flo5` predictions, and the 14.79 GB `flow_robustness.hdf5` are not duplicated in this bundle. The robust HDF5 hash and full-readback record remain in `inference_integrity.json`.

## Dataset declaration

| phase | data | exact use in this model |
|---|---|---|
| upstream DAv2 encoder pretraining | Depth Anything V2 ViT-S upstream data | Encoder weights are embedded in the downloaded WAFT parent. The exact upstream image subset and training log are not preserved locally; this stage cannot be independently reconstructed from this bundle. |
| WAFT parent pretraining/fine-tuning | TartanAir, FlyingChairs, FlyingThings3D, then Spring | Declared by the upstream checkpoint/config name `tar-c-t-spring-540p` and WAFT model zoo. The bundled parent is the official `waftv2-ckpts/dav2/spring.pth`; intermediate checkpoints and upstream optimizer logs were not locally preserved. |
| CAR-GR fine-tuning | public Spring training split | 28 fixed scenes listed in `experiments/stage2_5/train_scenes.txt`; 14,688 FW/BW, left/right pairs. Online paired clean/corrupt training uses the same flow GT, crop 540x960, severity 1-4, and 18 synthetic corruptions. `elastic_transform` and `glass_blur` are excluded from training. |
| checkpoint selection/validation | public Spring training split, held-out local scenes | Scenes `0002`, `0006`, `0015`, `0022`; 64 deterministic balanced scene x direction x side pairs. Metrics cover clean plus the same 18 synthetic corruptions. No test images or hidden GT are used. |
| official clean inference | public Spring test images | 10 scenes and 3,960 directional stereo pairs; no public GT. Produces the included clean HDF5. |
| RobustSpring inference | public RobustSpring test corruptions | The same 3,960 pairs under 20 corruptions, 79,200 predictions; no public flow GT. |

Dataset download references are retained in the included WAFT README. Main sources:

- Spring: https://spring-benchmark.org/
- RobustSpring/devkit: https://github.com/hmorimitsu/roco-spring-devkit
- TartanAir: https://theairlab.org/tartanair-dataset/
- FlyingChairs: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
- FlyingThings3D: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2

Expected local layout:

```text
data/spring/train/<scene>/{frame_left,frame_right,flow_FW_left,flow_FW_right,flow_BW_left,flow_BW_right}/...
data/spring/test/<scene>/{frame_left,frame_right}/...
data/robust_spring/<corruption>/test/<scene>/{frame_left,frame_right}/...
```

## Environment

```bash
conda env create -f environment.yml
conda activate car-gr-waft-step2000
```

The measured environment was Python 3.12.3, PyTorch 2.11.0+cu128, and CUDA 12.8. `xformers` was not installed; the included DAv2 implementation uses its working fallback.

## Training

Run from the extracted bundle root after placing the datasets at the paths above:

```bash
PYTHONPATH=. python experiments/car_vs_cargr.py --self-check

PYTHONPATH=. python experiments/car_vs_cargr.py \
  --branch CAR_GR \
  --output-root results/reproduction_car_gr \
  --steps 6000 \
  --height 540 --width 960 \
  --eval-samples 64 --eval-batch 8 \
  --workers 8 --cargr-batch 8
```

The fixed contract is seed 0, AdamW, LR 2.5e-5, weight decay 1e-5, cosine schedule with 5% warmup, BF16+TF32, effective batch 8, and evaluation at steps 0/500/1000/2000/3000/4500/6000. The script selects `best.pt` by minimum ProxyRbS and records the terminal step separately.

To reproduce the full CAR versus CAR-GR comparison, use the same arguments with `--run-all`; it runs the two branches serially.

## Evaluation

Checkpoint-only local evaluation:

```bash
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py self-check
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py evaluate --run RUN_A
```

The result is written to `experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/RUN_A/result.json`. This reproduces the disclosed Spring-dev clean and 18-corruption local proxy protocol. The historical selected values are:

- clean EPE: 2.02480743780023
- corrupt EPE: 2.727980523317088
- ProxyRbS: 0.97168343552713

These are local proxy metrics, not an official RobustSpring hidden-test score.

Official clean and RobustSpring HDF5 generation:

```bash
BUNDLE_ROOT=$(pwd)
PYTHONPATH=. python tools/official_transfer_probe.py \
  --stage both --batch-size 16 --writers 16 --no-finalize

(
  cd results/car_vs_cargr/official_transfer_probe
  ./subsampling_tools/flow_subsampling \
    "$BUNDLE_ROOT/outputs/car_gr_step2000_official/spring-robust/clean/test" \
    > clean_hdf5_generation.log 2>&1
  ./subsampling_tools/flow_robust_subsampling \
    "$BUNDLE_ROOT/outputs/car_gr_step2000_official/spring-robust" \
    > robust_hdf5_generation.log 2>&1
)

PYTHONPATH=. python tools/official_transfer_probe.py --finalize
```

The official benchmark metric itself requires the benchmark's hidden ground truth/server. This bundle reproduces the submitted predictions and HDF5 payload, but cannot locally reproduce a hidden server score.

## Resource requirements

| task | measured hardware | peak VRAM | wall-clock |
|---|---|---:|---:|
| CAR-GR fine-tuning, batch 8 | 1x NVIDIA A100-SXM4-80GB | 67,509 MiB observed process memory; 62,929.9 MiB PyTorch peak allocated recorded during the run | 26,754.5 s active logged time across two resume segments; artifact span at least 32,713.5 s (9 h 05 m), including checkpoint evaluations |
| canonical local evaluation, batch 4 | 1x NVIDIA A100-SXM4-80GB | 3,885.4 MiB PyTorch peak allocated in the isolated canonical run | 735.0 s for one 64-pair x 19-condition run |
| official-style forward, batch 16 | 1x NVIDIA A100-SXM4-80GB | 14,854.3 MiB allocated / 19,876 MiB reserved | 2.920 s for a no-write 16-pair batch; full clean inference log spans 2,088.54 s including interrupted/resumed attempts and `.flo5` writing |

Use an 80 GB GPU for the exact batch-8 training command. Smaller GPUs require lowering the micro-batch and accumulating to effective batch 8; that variant was not measured here. Parent WAFT/DAv2 upstream training hardware and wall-clock were not preserved, so no value is invented for them. HDF5 packing wall-clock was also not recorded separately.

## Validation and claim boundary

`results/car_vs_cargr/official_transfer_probe/inference_integrity.json` records 83,160 raw predictions, zero temporary files, successful official splitters, full HDF5 readback, and finite values. The clean submission is included. The robust HDF5 is omitted only for package size and can be checked against SHA256 `3a3cc280979401242cded05d4feca30da12446af8787ed655c15e7caadbd5363` after regeneration.
