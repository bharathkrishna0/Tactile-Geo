# Accuracy optimisation, iterations 1-3

Benchmark: `math100-v1` (100 generated images, exact ground truth recorded at
render time). Synthetic results only; the NCERT worksheets have no verified
ground truth and are not scored here. Model B is off in every run below, so
these numbers measure Model A alone.

Protocol:

- Tuning used only `evaluation/iterations/dev15.txt` (15 training-split images,
  3 per tier) and the 15-image validation split.
- The 15-image test split was run once on `main` (baseline) and once on the
  final iteration, after all tuning decisions were made.
- Every run is a fresh pipeline capture; all tables come from
  `python -m evaluation.report --rescore` on the stored captures, so baseline
  and optimised runs are scored by the same metric code.
- n = 15 per split. Differences of a few points are within the noise of a
  sample this small.

Runs (all under `evaluation/results/`):

| Run | Code |
|---|---|
| `baseline_v0-*` | `main` at 2e52760 |
| `iter1-*` | iteration 1 only (geometry) |
| `iter3c-*` | iterations 1-3 as merged in this PR |

Intermediate runs `iter3` and `iter3b` (not committed) were used for two
keep/revert decisions, described under "Decisions".

## What changed

Iteration 1 - geometry

- Closed shapes rebuilt from joined sides (`reconstruct_closed_shapes`):
  a single simple cycle of segments becomes a triangle / rectangle / polygon;
  ambiguous multi-loop structures stay as segments. Round shapes need at least
  9 vertices, so polygons are not turned into circles.
- Short tick marks crossing a long host line are recovered
  (`recover_ticks`) and marked `role: tick`; one-sided strokes are rejected.
  Simplification lengthens ticks to the touch minimum instead of deleting them.
- Line fragments inside OCR text boxes are dropped before vectorisation.

Iteration 2 - relationships

- New `VERTEX_OF` (point on a polygon corner) and `LIES_ON` (tick on its host).
- Parallel candidates need overlap and a bounded gap; perpendicular is inferred
  for line/polygon-side pairs.
- Right-angle markers are found at T-junctions as well as corners.
- Preprocessing restores thin strokes attached to bold lines when their ink is
  at least 60% as dark as the bold ink (`_attached_hairlines`). Without this,
  the opening step erased every right-angle marker on the benchmark.

Iteration 3 - OCR and label association

- Isolated single glyphs missed by EasyOCR detection are re-read with a
  restricted character set; repeated dash runs (dashed lines) are discarded.
- Label association uses evidence in order: nearby point, text continuing a
  tick, smallest enclosing closed shape, nearest outline. Each label now stores
  `association = {target_id, target_type, reason, confidence, needs_review}`
  and the benchmark capture records it.

Simplification

- When a sheet is over the 40-feature budget, long axis-aligned lines in a
  family of 5+ distinct parallel rows/columns (background grid or ruling) are
  dropped before other lines (`_grid_lines`).

## Results

### Validation split (n = 15)

| Metric (validation, n=15) | baseline_v0-validation | iter1-validation | iter3c-validation |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.218 | 0.232 | 0.260 |
| Object recall (strict) | 0.416 | 0.506 | 0.539 |
| Object F1 (strict) | 0.286 | 0.318 | 0.350 |
| Object F1 (structural: parts allowed) | 0.356 | 0.359 | 0.403 |
| Essential object recall, final tactile | 0.797 | 0.797 | 0.814 |
| Object type accuracy (matched) | 0.432 | 0.578 | 0.604 |
| Line endpoint error px, P50 | 1.267 | 1.284 | 1.259 |
| Line endpoint error px, P95 | 25 | 25 | 46 |
| Angle error deg, P50 | 0.089 | 0.126 | 0.126 |
| Polygon IoU, P50 | 0.992 | 0.987 | 0.985 |
| Circle centre error px, P50 | n/a | n/a | n/a |
| Point recall | 0.750 | 0.750 | 0.750 |
| Right-angle marker recall | 0.000 | 0.000 | 0.000 |
| Label detection recall | 0.398 | 0.398 | 0.576 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.539 | 0.539 | 0.461 |
| OCR CER (detected labels only) | 0.236 | 0.236 | 0.219 |
| OCR WER (all essential labels) | 0.729 | 0.729 | 0.597 |
| Label association accuracy | 0.217 | 0.217 | 0.651 |
| Braille end-to-end exact | 0.667 | 0.667 | 0.727 |
| Braille cell error rate | 0.277 | 0.277 | 0.244 |
| Braille reading-order concordance | 0.934 | 0.934 | 0.854 |
| Relationship recall (supported, final) | 0.136 | 0.136 | 0.182 |
| Relationship precision (mappable predictions) | 0.028 | 0.032 | 0.049 |
| Relationship F1 | 0.046 | 0.052 | 0.077 |
| Edges represented once (correct grouping) | 0.575 | 0.660 | 0.667 |
| Edge over-segmentation rate | 0.053 | 0.046 | 0.117 |
| Edge duplicate rate | 0.011 | 0.009 | 0.008 |
| Edges missed | 0.379 | 0.288 | 0.216 |
| Semantic Preservation Rate (end to end) | 0.259 | 0.259 | 0.400 |
| Simplification semantic preservation | 1.000 | 1.000 | 1.021 |
| Essential elements removed by simplification | 0 | 0 | 0 |
| Quantitative labels exact | 0.365 | 0.365 | 0.556 |
| Tactile QA pass rate | 0.933 | 0.933 | 0.933 |
| Critical problems exported (not blocked) | 0 | 0 | 0 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 11,442 | 11,805 | 11,145 |
| Total latency ms, P95 | 12,180 | 15,583 | 12,121 |

### dev15 tuning subset (n = 15)

| Metric (dev15, n=15) | baseline_v0-dev15 | iter1-dev15 | iter3c-dev15 |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.338 | 0.393 | 0.419 |
| Object recall (strict) | 0.411 | 0.571 | 0.509 |
| Object F1 (strict) | 0.371 | 0.465 | 0.460 |
| Object F1 (structural: parts allowed) | 0.397 | 0.476 | 0.496 |
| Essential object recall, final tactile | 0.582 | 0.595 | 0.595 |
| Object type accuracy (matched) | 0.587 | 0.719 | 0.702 |
| Line endpoint error px, P50 | 1.364 | 1.506 | 1.220 |
| Line endpoint error px, P95 | 73 | 33 | 86 |
| Angle error deg, P50 | 0.197 | 0.213 | 0.189 |
| Polygon IoU, P50 | 0.990 | 0.990 | 0.989 |
| Circle centre error px, P50 | 12 | 12 | 12 |
| Point recall | 0.549 | 0.649 | 0.649 |
| Right-angle marker recall | 0.000 | 0.000 | 0.429 |
| Label detection recall | 0.238 | 0.238 | 0.423 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.758 | 0.758 | 0.648 |
| OCR CER (detected labels only) | 0.228 | 0.228 | 0.214 |
| OCR WER (all essential labels) | 0.802 | 0.802 | 0.656 |
| Label association accuracy | 0.378 | 0.444 | 0.625 |
| Braille end-to-end exact | 0.711 | 0.711 | 0.762 |
| Braille cell error rate | 0.205 | 0.205 | 0.179 |
| Braille reading-order concordance | 0.908 | 0.908 | 0.924 |
| Relationship recall (supported, final) | 0.087 | 0.076 | 0.142 |
| Relationship precision (mappable predictions) | 0.079 | 0.072 | 0.187 |
| Relationship F1 | 0.083 | 0.074 | 0.162 |
| Edges represented once (correct grouping) | 0.419 | 0.516 | 0.521 |
| Edge over-segmentation rate | 0.034 | 0.027 | 0.071 |
| Edge duplicate rate | 0.090 | 0.073 | 0.071 |
| Edges missed | 0.521 | 0.409 | 0.393 |
| Semantic Preservation Rate (end to end) | 0.165 | 0.168 | 0.274 |
| Simplification semantic preservation | 0.939 | 0.940 | 0.959 |
| Essential elements removed by simplification | 3 | 3 | 2 |
| Quantitative labels exact | 0.200 | 0.200 | 0.320 |
| Tactile QA pass rate | 0.800 | 0.667 | 0.733 |
| Critical problems exported (not blocked) | 1 | 1 | 0 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 11,158 | 11,072 | 10,906 |
| Total latency ms, P95 | 13,971 | 14,928 | 11,592 |

### Held-out test split (n = 15, run once at the end)

| Metric (test, n=15) | baseline_v0-test | iter3c-test |
|---|---|---|
| Images failed (error/timeout) | 0 | 0 |
| Object precision (strict) | 0.421 | 0.456 |
| Object recall (strict) | 0.676 | 0.706 |
| Object F1 (strict) | 0.519 | 0.554 |
| Object F1 (structural: parts allowed) | 0.572 | 0.597 |
| Essential object recall, final tactile | 0.861 | 0.886 |
| Object type accuracy (matched) | 0.652 | 0.681 |
| Line endpoint error px, P50 | 1.338 | 1.337 |
| Line endpoint error px, P95 | 56 | 54 |
| Angle error deg, P50 | 0.198 | 0.188 |
| Polygon IoU, P50 | 0.974 | 0.982 |
| Circle centre error px, P50 | 0.000 | 0.000 |
| Point recall | 0.778 | 0.870 |
| Right-angle marker recall | 0.000 | 0.500 |
| Label detection recall | 0.263 | 0.399 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.720 | 0.692 |
| OCR CER (detected labels only) | 0.342 | 0.369 |
| OCR WER (all essential labels) | 0.812 | 0.759 |
| Label association accuracy | 0.294 | 0.560 |
| Braille end-to-end exact | 0.600 | 0.583 |
| Braille cell error rate | 0.236 | 0.274 |
| Braille reading-order concordance | 1.000 | 0.979 |
| Relationship recall (supported, final) | 0.092 | 0.146 |
| Relationship precision (mappable predictions) | 0.016 | 0.046 |
| Relationship F1 | 0.028 | 0.070 |
| Edges represented once (correct grouping) | 0.622 | 0.710 |
| Edge over-segmentation rate | 0.038 | 0.026 |
| Edge duplicate rate | 0.067 | 0.069 |
| Edges missed | 0.297 | 0.216 |
| Semantic Preservation Rate (end to end) | 0.225 | 0.298 |
| Simplification semantic preservation | 0.971 | 0.986 |
| Essential elements removed by simplification | 2 | 1 |
| Quantitative labels exact | 0.164 | 0.230 |
| Tactile QA pass rate | 1.000 | 1.000 |
| Critical problems exported (not blocked) | 1 | 1 |
| Images over 60 features | 0 | 0 |
| Total latency ms, P50 | 11,152 | 10,759 |
| Total latency ms, P95 | 11,839 | 11,562 |

## Decisions

- **Attached-hairline restoration, first version: reverted in place.**
  Restoring every thin attached stroke brought back light grid lines on
  `math_022` (14 -> 47 predicted elements, strict precision 0.357 -> 0.047).
  The kept version also requires ink darkness >= 60% of the bold strokes.
- **Curve-piece budget priority: discarded.** On `math_059` the density budget
  removed enough pieces of the essential curve that it no longer matched
  (a critical problem exported without a block). Prioritising segments that
  join end to end did not help, because Hough pieces of a curve overlap rather
  than join. Dropping grid lines first did fix it, and on the stored dev15
  captures it also reduced essential losses on `math_063` (3 -> 2).
- **Isolated-glyph OCR: kept.** Label detection recall rose on both dev15 and
  validation, and detected-label CER did not get worse.

## Regressions and limits

- **Edge over-segmentation rose** (validation 0.053 -> 0.117, dev15
  0.034 -> 0.071). More edges are found (missed edges 0.379 -> 0.216), but
  some are found as two pieces. Line endpoint P95 error also rose
  (validation 25 -> 46 px), driven by the same pieces.
- **Reading-order concordance fell on validation (0.934 -> 0.854).** It is
  computed over pairs of detected labels; detected labels went up, and one
  table image (`math_060`) went from 21 to 105 compared pairs, of which 79 are
  in order. On images whose pair count did not change, concordance is
  unchanged.
- **QA pass rate on dev15 fell (0.800 -> 0.733)**, while critical problems
  exported without a block went from 1 to 0. More sheets are blocked or
  flagged, and the one sheet that previously exported a critical problem no
  longer does.
- **On the test split, OCR on detected labels got slightly worse**
  (CER 0.342 -> 0.369, Braille exact 0.600 -> 0.583, cell error
  0.236 -> 0.274). These are averaged over detected labels only; 52% more
  labels are now detected, and the newly found ones are harder to read. Over
  all essential labels, CER and WER still improved (0.720 -> 0.692,
  0.812 -> 0.759). Test still has one sheet exporting a critical problem,
  as on `main`.
- **Right-angle marker recall on validation stays 0.** dev15 went from 0 to
  0.429; the validation markers are still missed.
- Relationship F1 remains low in absolute terms (validation 0.077).
- "Simplification semantic preservation" can exceed 1.0 (validation 1.021):
  collinear merges let a ground-truth object match after simplification that
  matched no single piece before. It is a ratio of found-after to found-before,
  not a capped rate.
- All numbers are on generated images. They are not evidence about real scans
  or photographs of worksheets.
