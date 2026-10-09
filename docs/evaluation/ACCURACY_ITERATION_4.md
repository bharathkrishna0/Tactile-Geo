# Accuracy optimisation, iteration 4: relationship evidence and drawn markers

Same benchmark and protocol as `ACCURACY_ITERATIONS_1-3.md`: `math100-v1`
(synthetic, exact ground truth), Model B off, tuning on `dev15` + validation
only, the 15-image test split run once after the candidate was chosen. All
tables come from `python -m evaluation.report --rescore`. n = 15 per split, so
differences of a few points are noise.

Runs: `iter3c-*` is the PR #10 base, `iter4d-*` is this change. An earlier
candidate (`iter4b`, not committed) added the markers without the
background-line rules below. It roughly doubled relationship recall but kept
validation precision at 0.106. Two of its images also timed out, so its recall
is not comparable to `iter4d`.

## What changed (all rule-based, Model A only)

Markers (`backend/app/services/markers.py`, `right_angles.py`)

- Measured ink is removed: the long edges Model A traced, plus OCR text boxes.
  Markers are searched for in the ink that remains.
- **Angle arcs**: residual ink at a nearly constant radius around a measured
  vertex counts as an arc when it covers a continuous run of angles. Both arc
  ends must lie on two different measured arms, and the arc's span must match
  the measured angle. Arcs at right-angle-marker vertices are skipped. Each arc
  becomes an `angle` element (`angle_marker: true`) with an `angle_association`
  to the point at its vertex (`ANGLE_AT`).
- **Parallel chevrons**: small mirrored stroke pairs sitting across a measured
  line are matched one-to-one (no residual piece is used twice) and counted
  (single vs double chevrons). A marker-backed `parallel_lines` relationship is
  emitted only when two lines carry the same count *and* measure within
  tolerance of parallel.
- **Right-angle markers**: in addition to corners and T-junctions, small
  L-shaped residual pieces whose legs lie along two measured perpendicular
  edges are accepted. L-shapes inside text boxes, or not attached to measured
  edges, are rejected.

Relationships (`geometry_relations.py`)

- **Equal length**: 1-3 tick marks near a segment's midpoint form a hash count.
  Two segments with the same count *and* measured lengths within tolerance get
  `equal_length`. Evenly spread tick families (axes, number lines) are ignored.
- **Background lines**: a line is treated as a table/grid rule if it is in a
  family of parallel grid lines, or if 3+ other lines meet it at right angles.
  Lines carrying 2+ tick marks are exempt (they are scales). Between background
  lines, only perpendicular corners (shared endpoint, e.g. the two axes) are
  stated. Parallel and crossing relationships that touch a background line are
  not stated.
- `intersects` now needs a proper crossing. Segments that only come within
  10 px of each other no longer produce a 0.6-confidence "appear to cross".

## Held-out test (n = 15, run once)

| Metric (test, n=15) | iter3c-test | iter4d-test |
|---|---|---|
| Images failed (error/timeout) | 0 | 0 |
| Object precision (strict) | 0.456 | 0.445 |
| Object recall (strict) | 0.706 | 0.676 |
| Object F1 (strict) | 0.554 | 0.537 |
| Object F1 (structural: parts allowed) | 0.597 | 0.592 |
| Essential object recall, final tactile | 0.886 | 0.835 |
| Object type accuracy (matched) | 0.681 | 0.667 |
| Line endpoint error px, P50 | 1.337 | 1.368 |
| Line endpoint error px, P95 | 54 | 39 |
| Angle error deg, P50 | 0.188 | 0.181 |
| Polygon IoU, P50 | 0.982 | 0.990 |
| Circle centre error px, P50 | 0.000 | 0.000 |
| Point recall | 0.870 | 0.889 |
| Right-angle marker recall | 0.500 | 0.667 |
| Angle-arc marker recall | 0.000 | 0.000 |
| Label detection recall | 0.399 | 0.399 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.692 | 0.692 |
| OCR CER (detected labels only) | 0.369 | 0.369 |
| OCR WER (all essential labels) | 0.759 | 0.759 |
| Label association accuracy | 0.560 | 0.600 |
| Braille end-to-end exact | 0.583 | 0.583 |
| Braille cell error rate | 0.274 | 0.274 |
| Braille reading-order concordance | 0.979 | 0.979 |
| Relationship recall (supported, final) | 0.145 | 0.321 |
| Relationship precision (mappable predictions) | 0.046 | 0.406 |
| Relationship F1 | 0.070 | 0.358 |
| Edges represented once (correct grouping) | 0.710 | 0.710 |
| Edge over-segmentation rate | 0.026 | 0.035 |
| Edge duplicate rate | 0.069 | 0.052 |
| Edges missed | 0.216 | 0.223 |
| Semantic Preservation Rate (end to end) | 0.298 | 0.339 |
| Simplification semantic preservation | 0.986 | 0.943 |
| Essential elements removed by simplification | 1 | 4 |
| Quantitative labels exact | 0.230 | 0.230 |
| Tactile QA pass rate | 1.000 | 1.000 |
| Critical problems exported (not blocked) | 1 | 1 |
| Images over 60 features | 0 | 0 |
| Total latency ms, P50 | 10,759 | 11,016 |
| Total latency ms, P95 | 11,562 | 11,895 |

## Validation (n = 15)

| Metric (validation, n=15) | iter3c-validation | iter4d-validation |
|---|---|---|
| Images failed (error/timeout) | 0 | 0 |
| Object precision (strict) | 0.260 | 0.262 |
| Object recall (strict) | 0.539 | 0.539 |
| Object F1 (strict) | 0.350 | 0.353 |
| Object F1 (structural: parts allowed) | 0.403 | 0.406 |
| Essential object recall, final tactile | 0.814 | 0.831 |
| Object type accuracy (matched) | 0.604 | 0.604 |
| Line endpoint error px, P50 | 1.259 | 1.267 |
| Line endpoint error px, P95 | 46 | 39 |
| Angle error deg, P50 | 0.126 | 0.069 |
| Polygon IoU, P50 | 0.985 | 0.986 |
| Circle centre error px, P50 | n/a | n/a |
| Point recall | 0.750 | 0.750 |
| Right-angle marker recall | 0.000 | 0.667 |
| Angle-arc marker recall | 0.000 | 1.000 |
| Label detection recall | 0.576 | 0.576 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.461 | 0.461 |
| OCR CER (detected labels only) | 0.219 | 0.219 |
| OCR WER (all essential labels) | 0.597 | 0.597 |
| Label association accuracy | 0.651 | 0.758 |
| Braille end-to-end exact | 0.727 | 0.727 |
| Braille cell error rate | 0.244 | 0.244 |
| Braille reading-order concordance | 0.854 | 0.871 |
| Relationship recall (supported, final) | 0.180 | 0.337 |
| Relationship precision (mappable predictions) | 0.049 | 0.345 |
| Relationship F1 | 0.077 | 0.341 |
| Edges represented once (correct grouping) | 0.667 | 0.660 |
| Edge over-segmentation rate | 0.117 | 0.118 |
| Edge duplicate rate | 0.008 | 0.017 |
| Edges missed | 0.216 | 0.222 |
| Semantic Preservation Rate (end to end) | 0.400 | 0.454 |
| Simplification semantic preservation | 1.021 | 1.021 |
| Essential elements removed by simplification | 0 | 0 |
| Quantitative labels exact | 0.556 | 0.556 |
| Tactile QA pass rate | 0.933 | 0.933 |
| Critical problems exported (not blocked) | 0 | 0 |
| Images over 60 features | 0 | 0 |
| Total latency ms, P50 | 11,145 | 11,334 |
| Total latency ms, P95 | 12,121 | 23,698 |

## dev15 tuning subset (n = 15)

| Metric (dev15, n=15) | iter3c-dev15 | iter4d-dev15 |
|---|---|---|
| Images failed (error/timeout) | 0 | 0 |
| Object precision (strict) | 0.419 | 0.416 |
| Object recall (strict) | 0.509 | 0.554 |
| Object F1 (strict) | 0.460 | 0.475 |
| Object F1 (structural: parts allowed) | 0.496 | 0.504 |
| Essential object recall, final tactile | 0.595 | 0.557 |
| Object type accuracy (matched) | 0.702 | 0.726 |
| Line endpoint error px, P50 | 1.220 | 1.494 |
| Line endpoint error px, P95 | 86 | 84 |
| Angle error deg, P50 | 0.189 | 0.182 |
| Polygon IoU, P50 | 0.989 | 0.989 |
| Circle centre error px, P50 | 12 | 12 |
| Point recall | 0.649 | 0.676 |
| Right-angle marker recall | 0.429 | 0.714 |
| Angle-arc marker recall | 0.000 | 1.000 |
| Label detection recall | 0.423 | 0.423 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.648 | 0.648 |
| OCR CER (detected labels only) | 0.214 | 0.214 |
| OCR WER (all essential labels) | 0.656 | 0.656 |
| Label association accuracy | 0.625 | 0.625 |
| Braille end-to-end exact | 0.762 | 0.762 |
| Braille cell error rate | 0.179 | 0.179 |
| Braille reading-order concordance | 0.924 | 0.920 |
| Relationship recall (supported, final) | 0.141 | 0.243 |
| Relationship precision (mappable predictions) | 0.187 | 0.469 |
| Relationship F1 | 0.161 | 0.320 |
| Edges represented once (correct grouping) | 0.521 | 0.521 |
| Edge over-segmentation rate | 0.071 | 0.078 |
| Edge duplicate rate | 0.071 | 0.069 |
| Edges missed | 0.393 | 0.376 |
| Semantic Preservation Rate (end to end) | 0.274 | 0.298 |
| Simplification semantic preservation | 0.959 | 0.898 |
| Essential elements removed by simplification | 2 | 5 |
| Quantitative labels exact | 0.320 | 0.320 |
| Tactile QA pass rate | 0.733 | 0.800 |
| Critical problems exported (not blocked) | 0 | 0 |
| Images over 60 features | 0 | 0 |
| Total latency ms, P50 | 10,906 | 11,470 |
| Total latency ms, P95 | 11,592 | 21,904 |

## Limitations

- **Parallel chevrons are not measured by this benchmark.** The generator
  draws parallel lines without chevrons, so the detector is covered only by
  synthetic unit tests (`backend/tests/test_markers.py`).
- **Equal-length ticks are not scored.** The ground truth has no
  `EQUAL_LENGTH` relationship, so this is also covered by unit tests only.
- **Angle arcs: only 8 in the whole benchmark** (2 in dev15, 1 in
  validation, 1 in test). The test arc was missed. The "1.000" recall on dev15
  and validation is based on 3 arcs.
- **Essential-object recall on the final sheet fell on test (0.886 -> 0.835).**
  Simplification removed 4 essential elements on test instead of 1, all on
  `math_061`. On that image the extra junction points and one right-angle
  marker pushed the sheet over the 40-feature budget. The budget drops points
  first, but it still had to remove 4 more short lines. Strict object F1 on test
  also fell slightly (0.554 -> 0.537).
- `ON` relationships (mostly ticks on axes) are often lost on the final sheet,
  because simplification removes short ticks.
- The background-line rule can hide a real relationship between lines that
  form part of a lattice, for example a triangle drawn on graph paper.
- Results are synthetic only. The NCERT worksheets have no verified ground
  truth.
