# CAR-GR for RoCo-Spring

Historical reproduction package for the RoCo-Spring Optical Flow Quantitative Track entry **RoCo-14 · ZXUv2.0**. The manuscript reports CAR-GR, a WAFT adaptation with paired clean/corrupted supervision, a frozen clean teacher, and an error-supervised gated residual head.

## Download v1.0.0

- [Complete historical reproduction bundle (1.05 GiB)](https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/download/v1.0.0/CAR-GR-WAFT-step2000-repro-20260828.tar.gz)
- [Release page](https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/tag/v1.0.0)
- [SHA256SUMS.txt](SHA256SUMS.txt)

The code and checkpoints are inside the attached bundle. GitHub's automatically generated **Source code (zip/tar.gz)** downloads contain only the repository contents, not the reproduction bundle.

Archive SHA-256:

```text
a2d8b2dc4fe3beecd983d15f6698f355e84db056fce06da101d7c20993e159f6
```

After downloading the archive and SHA256SUMS.txt into the same directory:

```bash
# macOS; on Linux use sha256sum -c SHA256SUMS.txt
shasum -a 256 -c SHA256SUMS.txt
tar -xzf CAR-GR-WAFT-step2000-repro-20260828.tar.gz
cd CAR-GR-WAFT-step2000-repro-20260828
shasum -a 256 -c BUNDLE_MANIFEST.sha256
```

## Included and external assets

The archive includes historical training/evaluation code, modified WAFT source, the parent and selected step-2000 checkpoints, environment specification, logs, local results, provenance records, and the clean submission HDF5. It also contains historical diagnostic records; their inclusion does not establish a causal benefit from gating.

Spring/RobustSpring datasets, raw flow predictions, and the large robust submission HDF5 are excluded. Obtain datasets separately from [Spring](https://spring-benchmark.org/) and the [RoCo-Spring devkit](https://github.com/hmorimitsu/roco-spring-devkit). Follow the bundled README for the required directory layout and official inference/packing commands.

The actual bundled WAFT source includes local changes relative to upstream commit `b152ff1cad1af8c185ee7b141997c48ff3334c87`; checking out that upstream commit alone is insufficient.

## Setup and local evaluation

Run from the extracted archive root after preparing the datasets:

```bash
conda env create -f environment.yml
conda activate car-gr-waft-step2000
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py self-check
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py evaluate --run RUN_A
```

The recorded environment uses Python 3.12.3, PyTorch 2.11.0+cu128 and CUDA 12.8. These are Linux/NVIDIA GPU instructions; downloading on macOS does not provide that execution environment. See the bundled README for hardware measurements.

Expected historical selected-checkpoint results on the fixed **64-pair development protocol**:

| Metric | Recorded value |
|---|---:|
| Local clean GT EPE | 2.02480743780023 |
| Local corrupted GT EPE | 2.727980523317088 |
| Local ProxyRbS | 0.97168343552713 |

These are development results used in selection, not independent test estimates or official prediction-disagreement scores. Official server scores require the benchmark's hidden ground truth.

## Historical training

```bash
PYTHONPATH=. python experiments/car_vs_cargr.py --self-check
PYTHONPATH=. python experiments/car_vs_cargr.py \
  --branch CAR_GR --output-root results/reproduction_car_gr \
  --steps 6000 --height 540 --width 960 \
  --eval-samples 64 --eval-batch 8 --workers 8 --cargr-batch 8
```

The historical schedule evaluates steps 0/500/1000/2000/3000/4500/6000; step 2000 was selected by local ProxyRbS. A new run's selected step and numerical outputs are not guaranteed to be bit-identical. The recorded batch-8 training used an A100 80 GB GPU.

## Known limitations and verification status

- The historical augmentation flips images and flow without correspondingly flipping the validity mask before cropping. Nonuniform masks may become misaligned. Its frequency and effect on the reported scores have not been measured. This release preserves the historical implementation; any correction should be released as a separately identified version.
- Improvements describe the selected full adaptation workflow. They do not isolate a spatial-gating causal effect; the selected checkpoint's direct residual effect is small in the local development evaluation.
- Package/download integrity checks do not constitute a fresh training or model-evaluation reproduction. No new training or evaluation was performed for this publication check.
- Upstream pretraining cannot be fully reconstructed from the retained records. Benchmark datasets must be downloaded separately.
- Third-party code, weights, and datasets retain their original license terms. This repository does not grant a new blanket license over bundled third-party assets; an explicit license for the authors' additions remains to be specified.

Selected checkpoint SHA-256: `2321390c920727f23c7ad28479b1aea63442e58dbb45a00cfea82c0247b7221f`.

This is a code/checkpoint release. It does not establish manuscript acceptance, final challenge ranking, or proceedings publication.
