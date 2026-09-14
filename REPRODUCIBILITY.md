# Reproduction scope and asset documentation

This document supplements the unchanged historical v1.0.0 archive. It describes the RoCo-Spring submission, not the later 20-pair or full80 studies.

## Model and intended use

CAR-GR adapts the bundled WAFT parent using paired clean/corrupted supervision, a frozen clean teacher, and an error-supervised gated residual head. DAv2 stays frozen; other specified WAFT modules and the head are trained jointly. The teacher and ground truth are not required at inference. The intended use is research on optical-flow robustness. Development results do not establish deployment safety or a causal advantage of spatial gating.

## Reproduction and verification routes

| Result | Bundled route |
|---|---|
| Selected CAR-GR checkpoint | `results/car_vs_cargr/CAR_GR/best.pt`; checkpoint-only evaluator in `experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py` |
| Parent baseline | Bundled `experiments/external_benchmark/weights/waft/extracted/waft_dav2_a2.pth`; retained `results/car_vs_cargr/baseline.json` |
| CAR/CAR-GR comparison | `experiments/car_vs_cargr.py --run-all` with the README's common training arguments; retained `results/car_vs_cargr/comparison.json` |
| Historical CAR-GR selection trajectory | Scheduled evaluation JSON files in `results/car_vs_cargr/CAR_GR/evaluations/` |
| Official predictions | `tools/official_transfer_probe.py` and the bundled subsampling executables; see the archive README for commands |
| Official scores | Benchmark result pages cited by the manuscript; the hidden server scores cannot be computed from locally available labels |

Run commands from the extracted archive root. Use the README's environment and data layout. The CAR-GR checkpoint-only evaluator loads the historical checkpoint at its fixed bundled path; evaluating a newly trained checkpoint requires deliberately updating that path. A complete historical CAR checkpoint trajectory is not retained, but the CAR training route and selected result are provided.

Required external images are Spring and RobustSpring. Dataset declarations, upstream lineage, hardware measurements, environment versions, and complete setup commands are in the archive README. The package provides a route from the supplied parent, not a reconstruction of all upstream pretraining.

## Historical hyperparameters and coordinate conventions

The source `experiments/car_vs_cargr.py` defines learning rate as 0.25 times its 1e-4 Spring reference, giving 2.5e-5. Seed 0, effective batch 8, AdamW settings, loss coefficients, gate-target constants, and the 6000-step schedule are fixed by the historical implementation. The retained records do not establish an exhaustive search or the original rationale for every constant. Checkpoint selection uses the fixed 64-pair local ProxyRbS.

Training flow and gate-target EPE use pixels on the augmented 540x960 output grid; target midpoint 0.5 and scale 0.15 have those units. Local development flow is restored to the GT grid with corresponding vector-component scaling before EPE is measured.

## Fixed development corruption strengths

Development calls `residual_experiment._condition_images`, then `fast_benchmark/benchmark.py::corrupt_pair`. Eight conditions dispatch to `stage1_tournament/stage1.py::_corrupt_image`; ten retained conditions use `benchmark.py::_corrupt_one`. Corruptions are applied before network-input resizing. Both frames use the same sample-ID/condition seed.

These are fixed implementations, not an average over training severity 1–4. Intensities below are normalized to [0,1] unless stated otherwise; spatial lengths refer to the original image grid.

| Condition | Historical setting |
|---|---|
| Brightness | Add 0.18 |
| Contrast | Multiply deviation from per-channel spatial mean by 0.45 |
| Saturation | Multiply HSV saturation by 0.25 |
| Gaussian blur | 9x9 kernel, sigma 2 |
| Defocus blur | Normalized disk, radius 5 |
| Motion blur | Length 15, angle uniform in [-45,45] degrees |
| Zoom blur | Average original and center crops at scales 1.03, 1.06, 1.09, 1.12 |
| Gaussian noise | Standard deviation 0.12 |
| Impulse noise | Shared spatial mask: probability 0.06 black and 0.06 white |
| Shot noise | Poisson(value * 12) / 12 |
| Speckle noise | value * Normal(0, 0.22) added to value |
| JPEG | Quality 25 |
| Pixelation | Downsample to floor(width/4), floor(height/4) with area interpolation; nearest-neighbor restore |
| Fog | Coarse noise at approximately 1/16 resolution, Gaussian sigma 1.5, cubic restore and min-max normalization; 0.55 image + 0.45 fog |
| Frost | Uniform noise smoothed at sigma 7 and min-max normalized; BGR tint (f,0.95f,0.75f); 0.65 image + 0.45 tint |
| Rain | max(1,H*W//5000) one-pixel streaks, displacement (6,28), value 0.8; 3x3/sigma-0.8 smoothing; 0.78 image + layer |
| Snow | Uniform-noise threshold 0.985, 5x5/sigma-0.8 smoothing; 0.72 image + 1.4 snow |
| Spatter | Uniform noise smoothed at sigma 5; mask clip((noise-0.52)*12,0,0.65); BGR mud (0.12,0.25,0.38) |

The released source defines clipping, rounding, borders, and random draws exactly. Elastic Transform and Glass Blur are excluded from the historical 18-condition development protocol. These local corruptions are not the official RobustSpring generation protocol.

## Limits, license status, and verification

The historical validity-mask flip issue described in the repository README remains present in v1.0.0. Its impact is unmeasured. Single-seed selected development gains and weather sensitivity remain limitations. Missing CAR-only and total-project compute records are not supplied by the CAR-GR timing measurements.

The archive was fully downloaded anonymously and matched its SHA-256; all 319 manifest entries passed. Static inspection checked the documented entry points, argument definitions, data paths, environment specification, parent/selected checkpoints, and retained records. This is not a new training or model-evaluation run.

Author-created code and documentation, including original additions in the historical archive, are now licensed under the repository's BSD-3-Clause LICENSE. Third-party source and model terms remain applicable; see THIRD_PARTY_NOTICES.md. The historical archive bytes remain unchanged.
