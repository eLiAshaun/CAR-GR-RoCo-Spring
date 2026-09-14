# CAR-GR historical step-2000 model

## Identity and use

- Entry: **RoCo-14 · ZXUv2.0**, RoCo-Spring Optical Flow Quantitative Track.
- Model: WAFT with a frozen DAv2 encoder and the CAR-GR gated residual head.
- Selected checkpoint: step 2000, from the historical seed-0, 6000-step adaptation schedule.
- Checkpoint SHA-256: `2321390c920727f23c7ad28479b1aea63442e58dbb45a00cfea82c0247b7221f`.
- Intended use: optical-flow robustness research and reproduction of the reported submission workflow. Benchmark performance does not establish deployment safety.

## Training and inference

CAR and CAR-GR start independently from the same bundled WAFT parent. Training uses paired clean/corrupted images with shared flow ground truth and a frozen parent teacher anchoring only the clean prediction. The DAv2 encoder stays frozen; the other specified WAFT modules and the CAR-GR head are trained jointly. The gate target uses detached pre-correction EPE; it is not an incremental corruption-damage label or calibrated uncertainty. Inference needs neither teacher nor ground truth.

Adaptation uses 28 public Spring training scenes and 18 synthetic corruptions at training severities 1–4. Selection uses 64 fixed pairs from four development scenes with fixed development corruption settings. The upstream parent was already trained on Spring. See [reproduction details](REPRODUCIBILITY.md) and the [historical README](docs/HISTORICAL_BUNDLE_README.md) for the complete data lineage, splits, environment and commands.

## Reported results

| Protocol / metric | Selected value |
|---|---:|
| Historical 64-pair local clean GT EPE | 2.02480743780023 |
| Historical 64-pair local corrupted GT EPE | 2.727980523317088 |
| Local ProxyRbS | 0.97168343552713 |
| Official Spring EPE, manuscript's 2026-09-08 snapshot | 0.298 |
| Official RobustSpring clean-to-corrupted prediction disagreement, same snapshot | 1.586 |
| Computed official RbS, same snapshot | 0.3063 |

Local EPE compares against flow GT; official disagreement compares clean and corrupted predictions. Local ProxyRbS and computed official RbS are different quantities. Official scores require the benchmark server; they are not recreated by the local evaluator. The official values here preserve the manuscript's dated snapshot and do not imply a current overall ranking.

## Limitations

- The selected local result is single-seed and uses the development set for selection.
- The comparison is between complete selected adaptation workflows, not an isolated causal test of the gate.
- The selected checkpoint's direct residual effect is small on the local development evaluation.
- Weather sensitivity remains substantial, and clean-subset improvements are uneven.
- The historical augmenter flips images and flow without flipping the validity mask before cropping. Nonuniform masks can become misaligned; frequency and impact are unmeasured. The released historical implementation retains this behavior.
- CAR-only and total-project compute are not fully recorded. Upstream pretraining is not reconstructible from this package alone.

## Availability and license

The [v1.0.0 archive](https://github.com/eLiAshaun/CAR-GR-RoCo-Spring/releases/tag/v1.0.0) contains the parent and selected checkpoints, historical source, environment, logs and records. Benchmark images and the large robust HDF5 are external. See [license scope](THIRD_PARTY_NOTICES.md) for author-code and third-party terms; no blanket BSD license is granted for pretrained weights.

Publication checks cover download integrity, the internal manifest, and static source/documentation consistency. No new model evaluation or training was run for this repository update.
