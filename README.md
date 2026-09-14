# CAR-GR for RoCo-Spring

**Clean-anchored optical-flow adaptation · RoCo-Spring Optical Flow Quantitative Track · RoCo-14 · ZXUv2.0**

CAR-GR adapts WAFT using paired clean/corrupted supervision, a frozen clean teacher, and an error-supervised gated residual head. This repository provides the historical implementation, the selected step-2000 reproduction package, and instructions for inspecting or reproducing the reported workflow.

[Download v1.0.0](https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/tag/v1.0.0) · [Reproduction details](REPRODUCIBILITY.md) · [Model card](MODEL_CARD.md) · [License scope](THIRD_PARTY_NOTICES.md)

## Quick start

Reproduction uses the **complete release bundle**, which includes checkpoints and retained records. A Git clone provides browsable source and documentation; it excludes large assets and run outputs. GitHub's automatic “Source code” archives also exclude the attached reproduction bundle.

Download and verify the complete package with Python 3.12 or newer:

```bash
git clone https://github.com/eLiAshaun/CAR-GR-RoCo-Spring.git
cd CAR-GR-RoCo-Spring
python3 tools/download_bundle.py
cd reproduction_artifacts/CAR-GR-WAFT-step2000-repro-20260828
conda env create -f environment.yml
conda activate car-gr-waft-step2000
```

Alternatively, download the [1.05 GiB archive](https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/download/v1.0.0/CAR-GR-WAFT-step2000-repro-20260828.tar.gz) and [SHA256SUMS.txt](SHA256SUMS.txt) manually into the same directory:

```bash
shasum -a 256 -c SHA256SUMS.txt  # Linux: sha256sum -c SHA256SUMS.txt
tar -xzf CAR-GR-WAFT-step2000-repro-20260828.tar.gz
cd CAR-GR-WAFT-step2000-repro-20260828
shasum -a 256 -c BUNDLE_MANIFEST.sha256
```

Archive SHA-256: `a2d8b2dc4fe3beecd983d15f6698f355e84db056fce06da101d7c20993e159f6`.

## Prepare data

Obtain [Spring](https://spring-benchmark.org/) and, for official corrupted-input inference, [RobustSpring / RoCo-Spring devkit](https://github.com/hmorimitsu/roco-spring-devkit) separately. Under the **extracted bundle root**, use:

```text
data/spring/train/<scene>/{frame_left,frame_right,flow_FW_left,flow_FW_right,flow_BW_left,flow_BW_right}/...
data/spring/test/<scene>/{frame_left,frame_right}/...
data/robust_spring/<corruption>/test/<scene>/{frame_left,frame_right}/...
```

All following model commands run from the extracted bundle root. The recorded environment uses Linux, Python 3.12.3, PyTorch 2.11.0+cu128 and CUDA 12.8. Model execution requires an appropriate NVIDIA GPU; downloading the package on macOS does not supply that environment.

## Evaluate the selected checkpoint

```bash
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py self-check
PYTHONPATH=. python experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/canonical_eval.py evaluate --run RUN_A
```

The evaluator loads the historical checkpoint at its fixed bundled path. The historical **64-pair local development** values are:

| Metric | Value |
|---|---:|
| Clean GT EPE | 2.02480743780023 |
| Corrupted GT EPE | 2.727980523317088 |
| Local ProxyRbS | 0.97168343552713 |

These development results were used in selection. They are not independent test estimates or official prediction-disagreement scores. See the [model card](MODEL_CARD.md) for the manuscript's dated official results.

## Train the adaptation

```bash
PYTHONPATH=. python experiments/car_vs_cargr.py --self-check
PYTHONPATH=. python experiments/car_vs_cargr.py \
  --branch CAR_GR --output-root results/reproduction_car_gr \
  --steps 6000 --height 540 --width 960 \
  --eval-samples 64 --eval-batch 8 --workers 8 --cargr-batch 8
```

To run both independently initialized CAR and CAR-GR branches, replace `--branch CAR_GR` with `--run-all` and choose a separate output directory. The historical schedule evaluates steps 0/500/1000/2000/3000/4500/6000; step 2000 was selected by local ProxyRbS. A new run is not guaranteed to select the same step or produce bit-identical values.

The recorded effective-batch-8 training used one A100 80 GB GPU. See the [historical README](docs/HISTORICAL_BUNDLE_README.md) for measured memory/time, full official prediction and HDF5-generation commands, and dataset lineage.

## Source and artifact layout

| Path | Contents |
|---|---|
| `experiments/car_vs_cargr.py` | Historical adaptation and comparison entry point |
| `experiments/CAR_GR_EVAL_PROTOCOL_LOCK_V1/` | Selected-checkpoint local evaluator |
| `experiments/residual_screen/` | Historical head/evaluation helpers imported by the entry |
| `experiments/external_benchmark/sources/WAFT/` | Modified upstream source with retained notices |
| `tools/official_transfer_probe.py` | Historical official-prediction runner |
| `tools/download_bundle.py` | New download, SHA-256 and archive-manifest verification helper |
| `environment.yml` | Historical model execution environment |
| Release attachment | Checkpoints, complete historical source, logs, retained metrics and clean HDF5 |

Historical source/configuration copies are listed in [SOURCE_MANIFEST.sha256](SOURCE_MANIFEST.sha256). The archive-only file list is in [docs/ARCHIVE_ONLY_FILES.txt](docs/ARCHIVE_ONLY_FILES.txt). The complete original bundle manifest is retained in `docs/HISTORICAL_BUNDLE_MANIFEST.sha256`; run it **inside the extracted bundle**, not against this Git checkout.

## Known limits

The historical validity-mask flip issue remains present; the effect on reported results is unmeasured. Gains belong to the selected full adaptation workflow and do not isolate a gating causal effect. Direct residual effects are small locally, weather sensitivity remains, and clean-subset improvements are uneven. Detailed limitations appear in the [model card](MODEL_CARD.md).

Download, manifest and static consistency checks are documented; no new training or model evaluation was run for this repository update. Later 20-pair/full80 studies are outside this repository's submission-report scope. Some historical helper files and diagnostic records are retained for provenance rather than offered as new workshop contributions.

## License and citation

Author-created code and documentation are licensed under [BSD-3-Clause](LICENSE). Third-party code, model weights and datasets retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This grant covers the authors' original additions in the unchanged historical archive as well.

Use [CITATION.cff](CITATION.cff) to cite the versioned software package. No proceedings DOI, acceptance, or final challenge ranking is asserted. Please report reproduction issues with the command and environment information described in [CONTRIBUTING.md](CONTRIBUTING.md).
