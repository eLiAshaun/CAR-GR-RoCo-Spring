# License scope and third-party notices

The root [LICENSE](LICENSE) grants BSD-3-Clause rights for the CAR-GR authors' original code contributions and repository documentation. This includes the authors' original additions distributed in the historical v1.0.0 archive. It does not replace notices or licenses applicable to incorporated third-party material, pretrained weights, or datasets. The archive remains byte-for-byte unchanged; this repository records the additional author-code grant.

| Asset | Applicable notice / source |
|---|---|
| WAFT source and incorporated code | [Bundled BSD-3-Clause notice](experiments/external_benchmark/sources/WAFT/LICENSE); preserve individual file notices as well |
| DINOv3 material carried in the upstream source tree | [Bundled DINOv3 License](experiments/external_benchmark/sources/WAFT/thirdparty/dinov3/LICENSE.md); the CAR-GR entry uses DAv2, but the distributed DINOv3 files retain their own terms |
| DAv2-Small pretrained component | [Depth Anything V2 upstream model/license information](https://github.com/DepthAnything/Depth-Anything-V2); the Small model is described upstream under Apache-2.0 |
| Bundled WAFT parent and derived CAR-GR checkpoint | Retain applicable upstream weight/component terms; the root author-code license is not a blanket relicensing of these weights |
| Spring and RobustSpring images | Download separately from the benchmark sources; the manuscript records CC BY 4.0 dataset terms |
| Other dependencies | Their respective package licenses; versions are recorded in environment.yml |

The copied historical source files retain their original headers. No claim is made that every dependency or bundled asset is licensed under BSD-3-Clause. This notice adds no new restrictions to assets already covered by another license.
