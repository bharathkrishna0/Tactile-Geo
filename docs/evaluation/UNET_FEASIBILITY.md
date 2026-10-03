# U-Net feasibility pilot (CPU, synthetic masks)

**Decision: KEEP AS EXPERIMENTAL.** Do not put it in the production pipeline.
On the generated benchmark, a small CPU U-Net mask made Model A's
vectorisation cleaner and raised object F1. But every number here comes from
synthetic images and synthetic masks drawn by the same generator, real-world
segmentation quality is unmeasured, it adds CPU latency, and on one test image
it caused a QA block.

## Setup

| | |
|---|---|
| Code | `evaluation/unet/{model,data,train,infer,seg_eval}.py` |
| Weights | `evaluation/unet/weights/unet_small.pt` (1.9 MB), training log `unet_small.history.json` |
| Architecture | 4-level U-Net, base 16 channels, 482,449 parameters, 1-channel input, sigmoid output |
| Target | `evaluation/dataset/masks/*.png`: pixels of diagram strokes, drawn at render time (text excluded) |
| Data | train split only (70 images), 256 px random crops, batch 8, seed 0 |
| Training | 1,500 steps on CPU, 3,225 s (54 min). Best validation Dice at step 1,000, checkpoint kept from there |
| Selection | validation split (15) used for checkpoint selection, so validation numbers are optimistic. Test split (15) was untouched until the final runs |
| Hardware | CPU only, no GPU |

Training curve (validation Dice): 0.750 (step 250), 0.937 (500), 0.940 (750),
**0.949 (1,000)**, 0.942 (1,250), 0.944 (1,500).

## 1. Segmentation against the synthetic masks

`python -m evaluation.unet.seg_eval`, result file
`evaluation/results/unet_segmentation.json`. Pixel metrics, threshold 0.5,
pooled over the split. "Model A binary" is the existing
`preprocess_image()` output scored against the same masks. It is not trained
for this target and also keeps text, so its precision is expected to be lower.

| Split | Method | Dice | IoU | Precision | Recall | Median latency |
|---|---|---:|---:|---:|---:|---:|
| validation | U-Net | 0.949 | 0.903 | 0.932 | 0.966 | 2,785 ms |
| validation | Model A binary | 0.591 | 0.420 | 0.589 | 0.594 | 77 ms |
| **test** | **U-Net** | **0.879** | **0.785** | 0.826 | 0.940 | 2,970 ms |
| test | Model A binary | 0.569 | 0.397 | 0.490 | 0.678 | 70 ms |

The validation-to-test drop (0.949 to 0.879) is the clearest sign of
overfitting to 70 training images.

## 2. Downstream: does the mask improve the tactile output?

Same 15 held-out test images, same Model A, OCR, simplification, QA and SVG
code. Only the binary image handed to vectorisation changes.

* `model_a_v2-test-t600`: Model A as is.
* `model_a_v2+unet-test`: Model A's binary, kept only where the dilated U-Net
  mask is on (`combine="intersect"`).
* `unet_only-test`: the U-Net mask replaces Model A's binary.
  **Stopped on request after 11 of 15 images**, so it is compared on those 11
  only.

### All 15 test images

| Metric | Model A | Model A + U-Net |
|---|---:|---:|
| Object precision (strict) | 0.421 | **0.613** |
| Object recall (strict) | 0.676 | 0.667 |
| Object F1 (strict) | 0.519 | **0.638** |
| Object F1 (structural) | 0.572 | **0.705** |
| Essential object recall, final tactile | 0.861 | **0.911** |
| Object type accuracy | 0.652 | 0.642 |
| Line endpoint error P50 / P95 (px) | 1.34 / 56 | 1.63 / 67 |
| Polygon IoU P50 | 0.974 | 0.981 |
| Relationship recall | 0.109 | 0.127 |
| Essential elements removed by simplification | 2 | 0 |
| Label detection recall / OCR CER | 0.263 / 0.720 | 0.263 / 0.720 (OCR does not use the mask) |
| Tactile QA pass rate | **1.000** | 0.933 (math_042 blocked: feature_below_minimum_size) |
| Total latency P50 / P95 | 37 s / 176 s | 59 s / 487 s |

### The 11 images `unet_only-test` reached

| Metric | Model A | U-Net only | Model A + U-Net |
|---|---:|---:|---:|
| Object F1 (strict) | 0.614 | 0.719 | **0.725** |
| Object F1 (structural) | 0.639 | 0.747 | **0.769** |
| Essential recall, final | 0.846 | 0.885 | **0.923** |
| QA pass rate | **1.000** | 0.909 | 0.909 |
| Latency P50 | 37 s | 61 s | 74 s |

By difficulty (Model A + U-Net, n=3 each, so treat as indicative only):
strict F1 0.875 easy, 0.609 moderate, 0.846 hard, 0.660 very hard, 0.381
adversarial.

Per image (`failure_analysis_test.json`), U-Net lowered the failure score on
5 of 15 images (math_034, 047, 062, 076, 083), raised it on 1 (math_042) and
left 9 unchanged.

## 3. CPU compatibility and cost

* Runs on CPU with stock PyTorch, which is already a backend dependency
  through EasyOCR. Weights are 1.9 MB, and no GPU or new package is needed.
* Inference takes about 3 s per image at benchmark resolution (median), against
  70 ms for the existing binarisation.
* **Latency caveat:** the downstream runs shared the CPU with other benchmark
  runs, so the end-to-end P50/P95 differences above are inflated and not a
  clean measurement. The isolated per-image mask cost is the 3 s figure.

## 4. Why not ADOPT

1. **Synthetic only.** Masks and images come from one generator. The U-Net
   may have learned that generator's stroke styles and fonts. No real
   worksheet has a pixel mask, so real-world segmentation is NOT MEASURED. The
   NCERT slice was not run with U-Net.
2. **Small, overfit training set:** 70 images, with test Dice 0.07 below
   validation.
3. **One new QA block** on the held-out set, where Model A alone had none.
4. **It does not touch the biggest error source.** Label recall and OCR are
   unchanged, and OCR is the main failure cause (74 of 100 images).

## 5. What would change the decision

* Pixel masks for 20 or more real worksheets, which a teacher can trace in
  an image editor, and U-Net Dice on them close to the synthetic test value.
* Re-running Model A + U-Net on the NCERT slice without new QA blocks.
* Isolated latency of 3 s or less per page on the target deployment CPU.
* Gating it behind a setting (e.g. `UNET_MASK=off|intersect`) with Model A
  remaining the default and the geometry authority.

## Reproduce

```bash
python -m evaluation.unet.train --steps 1500 --base 16 --seed 0
python -m evaluation.unet.seg_eval
python -m evaluation.run_benchmark --split test --model-a --unet --run-id model_a_v2+unet-test --image-timeout 600
python -m evaluation.run_benchmark --split test --unet --run-id unet_only-test --image-timeout 600
```
