# Accuracy optimisation, V1: text-first OCR and figure regions

Same benchmark and protocol as `ACCURACY_ITERATION_5.md`: `math100-v1`
(synthetic, exact ground truth), Model B off, keep/revert decisions on `dev15`
(`evaluation/iterations/dev15.txt`) + validation only, the 15-image test split
run **once**, after the candidate was fixed. Tables come from
`python -m evaluation.report --split all --rescore`. n = 15 per split, so
differences of a few points are noise. The NCERT slice has no verified ground
truth, so its accuracy is **NOT MEASURED**; it only gets annotation-free checks.
Nothing here is evidence about real tactile readers.

Runs:

- `v0-*`: `main` at the V0 baseline commit. Its numbers equal `iter5d` from
  iteration 5 (masking off). The earlier `iter5f` figures were the masking
  ablation, which is why they differed.
- `v1c-*`: region guard + OCR fusion changes.
- `v1d-*` / `v1d-test`: the candidate (v1c + shape names are not point names +
  the stricter dark-ink rule for small figures).
- `iter5g-test`: the held-out run of iteration 5 (same code path as V0) used as
  the test baseline, so the test split was not re-run for V0.

Latency: v1c and v1d runs shared the CPU with the NCERT runs, so latency is not
comparable between runs here.

## What changed

- **Small figures on worksheet pages** (`diagram_region.py`). A component that
  is short (below 8% of the page) used to be ignored as text. It is now kept as
  a figure when it is both tall and wide (>= 3.5% of the short side) and at
  least 95% of its pixels are dark ink. A pale watermark letter, a prose line,
  and prose letters joined by a pale watermark stroke are rejected (one
  regression test each). The 95% rule came from the NCERT pages: with a median
  test, prose touching the NCERT watermark was admitted on 3 pages and
  `ncert_03` became BLOCKED (a 1 mm noise polygon from the prose). Measured on
  the 20 NCERT pages, the final rule finds exactly the same regions as `main`.
- **Invalid OCR readings no longer hide glyphs** (`ocr.py`). Page readings
  such as `7"` or `€C` are dropped by `postprocess_detections` anyway, but they
  used to mark their box as taken so the isolated-glyph reader skipped the
  letter underneath. The same `VALID_TEXT_PATTERN` now filters them first.
- **OCR fusion** (`ocr_fusion.py`). A two-letter reading of one letter in both
  cases (`Cc`, `uU`) counts as that capital only when the other engine agrees.
  A glyph inside a short reading (`R` in `Ra`) counts only when the other
  engine read exactly that glyph. Stroke-like glyphs (`1 I i l | -`) never
  confirm anything. A single engine reading `Cc` alone is still rejected.
- **Shape names are not point names** (`label_association.py`,
  `diagram_analysis.py`). A word of 3+ letters (`hexagon`, `square`,
  `Mathematics`) no longer creates a point at a nearby vertex or claims one, so
  it can be linked to the shape.
- **Per-engine OCR metrics** (`evaluation/metrics.py`, `aggregate.py`,
  `report.py`). For labels from EasyOCR, Tesseract, and both: count, precision
  (share on a ground-truth label) and exact-read rate. Labels are traced to
  their raw OCR reading by text and box origin.

## dev15 (n = 15)

| Metric (all, n=15) | v0-dev15 | v1c-dev15 | v1d-dev15 |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.416 | 0.440 | 0.440 |
| Object recall (strict) | 0.554 | 0.589 | 0.589 |
| Object F1 (strict) | 0.475 | 0.504 | 0.504 |
| Object F1 (structural: parts allowed) | 0.504 | 0.532 | 0.532 |
| Essential object recall, final tactile | 0.557 | 0.633 | 0.633 |
| Object type accuracy (matched) | 0.726 | 0.742 | 0.742 |
| Point recall | 0.676 | 0.838 | 0.838 |
| Right-angle marker recall | 0.714 | 0.857 | 0.857 |
| Angle-arc marker recall | 1.000 | 1.000 | 1.000 |
| Label detection recall | 0.476 | 0.540 | 0.540 |
| Label detection precision | 0.851 | 0.865 | 0.865 |
| Labels from easyocr | 70 | 74 | 74 |
| Precision of easyocr labels | 0.757 | 0.770 | 0.770 |
| Exactly read, easyocr labels | 0.500 | 0.473 | 0.473 |
| Labels from tesseract | 5 | 4 | 4 |
| Precision of tesseract labels | 1.000 | 1.000 | 1.000 |
| Exactly read, tesseract labels | 1.000 | 1.000 | 1.000 |
| Labels from easyocr+tesseract | 39 | 48 | 48 |
| Precision of easyocr+tesseract labels | 1.000 | 1.000 | 1.000 |
| Exactly read, easyocr+tesseract labels | 0.872 | 0.896 | 0.896 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.611 | 0.556 | 0.556 |
| OCR CER (detected labels only) | 0.192 | 0.212 | 0.212 |
| OCR WER (all essential labels) | 0.604 | 0.566 | 0.566 |
| Label association accuracy | 0.644 | 0.637 | 0.647 |
| Braille end-to-end exact | 0.811 | 0.794 | 0.794 |
| Braille cell error rate | 0.159 | 0.167 | 0.167 |
| Braille reading-order concordance | 0.930 | 0.926 | 0.926 |
| Relationship recall (supported, final) | 0.265 | 0.308 | 0.308 |
| Relationship precision (mappable predictions) | 0.490 | 0.540 | 0.540 |
| Relationship F1 | 0.344 | 0.392 | 0.392 |
| Edges represented once (correct grouping) | 0.521 | 0.602 | 0.602 |
| Edge over-segmentation rate | 0.078 | 0.066 | 0.066 |
| Edge duplicate rate | 0.069 | 0.103 | 0.103 |
| Edges missed | 0.376 | 0.269 | 0.269 |
| Semantic Preservation Rate (end to end) | 0.332 | 0.375 | 0.376 |
| Simplification semantic preservation | 0.898 | 0.909 | 0.909 |
| Essential elements removed by simplification | 5 | 5 | 5 |
| Quantitative labels exact | 0.333 | 0.347 | 0.347 |
| Tactile QA pass rate | 0.800 | 0.800 | 0.800 |
| Critical problems exported (not blocked) | 0 | 0 | 0 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 11,740 | 9,217 | 9,233 |
| Total latency ms, P95 | 14,765 | 14,652 | 15,633 |

## Validation (n = 15)

| Metric (all, n=15) | v0-validation | v1c-validation | v1d-validation |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.262 | 0.258 | 0.258 |
| Object recall (strict) | 0.539 | 0.539 | 0.539 |
| Object F1 (strict) | 0.353 | 0.349 | 0.349 |
| Object F1 (structural: parts allowed) | 0.406 | 0.401 | 0.401 |
| Essential object recall, final tactile | 0.831 | 0.831 | 0.831 |
| Object type accuracy (matched) | 0.604 | 0.604 | 0.604 |
| Point recall | 0.750 | 0.806 | 0.806 |
| Right-angle marker recall | 0.667 | 0.667 | 0.667 |
| Angle-arc marker recall | 1.000 | 1.000 | 1.000 |
| Label detection recall | 0.619 | 0.627 | 0.627 |
| Label detection precision | 0.915 | 0.905 | 0.905 |
| Labels from easyocr | 56 | 56 | 56 |
| Precision of easyocr labels | 0.929 | 0.911 | 0.911 |
| Exactly read, easyocr labels | 0.589 | 0.571 | 0.571 |
| Labels from tesseract | 2 | 2 | 2 |
| Precision of tesseract labels | 1.000 | 1.000 | 1.000 |
| Exactly read, tesseract labels | 1.000 | 1.000 | 1.000 |
| Labels from easyocr+tesseract | 24 | 26 | 26 |
| Precision of easyocr+tesseract labels | 0.875 | 0.885 | 0.885 |
| Exactly read, easyocr+tesseract labels | 0.833 | 0.846 | 0.846 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.441 | 0.437 | 0.437 |
| OCR CER (detected labels only) | 0.213 | 0.211 | 0.211 |
| OCR WER (all essential labels) | 0.562 | 0.556 | 0.556 |
| Label association accuracy | 0.747 | 0.750 | 0.778 |
| Braille end-to-end exact | 0.747 | 0.750 | 0.750 |
| Braille cell error rate | 0.236 | 0.234 | 0.234 |
| Braille reading-order concordance | 0.868 | 0.870 | 0.870 |
| Relationship recall (supported, final) | 0.337 | 0.360 | 0.360 |
| Relationship precision (mappable predictions) | 0.345 | 0.356 | 0.356 |
| Relationship F1 | 0.341 | 0.358 | 0.358 |
| Edges represented once (correct grouping) | 0.660 | 0.667 | 0.667 |
| Edge over-segmentation rate | 0.118 | 0.117 | 0.117 |
| Edge duplicate rate | 0.017 | 0.017 | 0.017 |
| Edges missed | 0.222 | 0.216 | 0.216 |
| Semantic Preservation Rate (end to end) | 0.474 | 0.484 | 0.489 |
| Simplification semantic preservation | 1.021 | 1.021 | 1.021 |
| Essential elements removed by simplification | 0 | 0 | 0 |
| Quantitative labels exact | 0.587 | 0.587 | 0.587 |
| Tactile QA pass rate | 0.867 | 0.867 | 0.867 |
| Critical problems exported (not blocked) | 0 | 0 | 0 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 11,886 | 9,029 | 12,124 |
| Total latency ms, P95 | 16,841 | 13,725 | 15,397 |

## Decision

Keep v1d. On both tuning splits: label recall up (dev 0.476 -> 0.540,
validation 0.619 -> 0.627), association up (0.644 -> 0.647, 0.747 -> 0.778),
relationship F1 up (0.344 -> 0.392, 0.341 -> 0.358), essential-object recall
up on dev15 (0.557 -> 0.633) and unchanged on validation, QA pass rate
unchanged.

Costs on the tuning splits:

- Validation label precision 0.915 -> 0.905 and EasyOCR exact-read rate
  0.589 -> 0.571 (dev 0.500 -> 0.473).
- Validation object F1 (strict) 0.353 -> 0.349.
- dev15 edge duplicate rate 0.069 -> 0.103: the newly kept small figures bring
  their edges in twice on some images. V2 (duplicate and fragment cleanup).
- dev15 Braille end-to-end exact 0.811 -> 0.794.

## Held-out test (n = 15, run once)

| Metric (all, n=15) | iter5g-test | v1d-test |
|---|---|---|
| Images failed (error/timeout) | 0 | 0 |
| Object precision (strict) | 0.448 | 0.448 |
| Object recall (strict) | 0.676 | 0.676 |
| Object F1 (strict) | 0.539 | 0.539 |
| Object F1 (structural: parts allowed) | 0.594 | 0.594 |
| Essential object recall, final tactile | 0.835 | 0.848 |
| Object type accuracy (matched) | 0.667 | 0.667 |
| Point recall | 0.889 | 0.889 |
| Right-angle marker recall | 0.667 | 0.667 |
| Angle-arc marker recall | 0.000 | 0.000 |
| Label detection recall | 0.474 | 0.481 |
| Label detection precision | 0.863 | 0.865 |
| Labels from easyocr | 67 | 66 |
| Precision of easyocr labels | 0.866 | 0.864 |
| Exactly read, easyocr labels | 0.284 | 0.288 |
| Labels from tesseract | 8 | 8 |
| Precision of tesseract labels | 0.875 | 0.875 |
| Exactly read, tesseract labels | 0.500 | 0.500 |
| Labels from easyocr+tesseract | 20 | 22 |
| Precision of easyocr+tesseract labels | 0.850 | 0.864 |
| Exactly read, easyocr+tesseract labels | 0.650 | 0.636 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.664 | 0.661 |
| OCR CER (detected labels only) | 0.358 | 0.355 |
| OCR WER (all essential labels) | 0.712 | 0.706 |
| Label association accuracy | 0.567 | 0.590 |
| Braille end-to-end exact | 0.621 | 0.627 |
| Braille cell error rate | 0.263 | 0.260 |
| Braille reading-order concordance | 0.947 | 0.949 |
| Relationship recall (supported, final) | 0.321 | 0.336 |
| Relationship precision (mappable predictions) | 0.409 | 0.409 |
| Relationship F1 | 0.360 | 0.369 |
| Edges represented once (correct grouping) | 0.710 | 0.710 |
| Edge over-segmentation rate | 0.035 | 0.035 |
| Edge duplicate rate | 0.052 | 0.052 |
| Edges missed | 0.223 | 0.223 |
| Semantic Preservation Rate (end to end) | 0.363 | 0.375 |
| Simplification semantic preservation | 0.943 | 0.957 |
| Essential elements removed by simplification | 4 | 3 |
| Quantitative labels exact | 0.279 | 0.279 |
| Tactile QA pass rate | 1.000 | 0.933 |
| Critical problems exported (not blocked) | 1 | 1 |
| Images over 60 features | 0 | 0 |
| Total latency ms, P50 | 5,261 | 8,514 |
| Total latency ms, P95 | 12,961 | 13,194 |

- **Better:** essential-object recall 0.835 -> 0.848, label recall
  0.474 -> 0.481, association 0.567 -> 0.590, relationship F1
  0.360 -> 0.369, semantic preservation 0.363 -> 0.375, essential elements
  removed by simplification 4 -> 3. All small.
- **Worse:** QA pass rate 1.000 -> 0.933. `math_076` is now BLOCKED for a
  0.8 mm noise polygon under the page header. Its regions and OCR are identical
  to the baseline; the header words (`Mathematics`, `Name:`) no longer create
  points (15 -> 9 points), and the noise polygon survives. The gate caught it,
  so it is not exported silently. It was not fixed, because fixing it would
  tune on the test split.
- 1 test sheet still exports with a critical problem, as before.

## NCERT slice (20 pages, no ground truth)

| Check | V0 (`real_ncert`) | V1 (`real_ncert_v1d`) |
|---|---|---|
| Pages failed | 0 | 0 |
| Valid SVGs | 20 | 20 |
| QA pass rate | 0.70 | 0.70 |
| Labels found, mean / median | 15.7 / 14 | 16.15 / 14 |
| Blocking reasons | braille_on_line 5, feature_below_minimum_size 1, density_reduction_requires_review 1, element_outside_printable_area 1 | same |

8 pages gained labels (for example `A` x4 on `ncert_02`, `E` x2 on
`ncert_11`, `X` on `ncert_14`); `ncert_08` now reads `O` where it read `0`.
Whether these are correct is not measured until the annotations are verified.

## Remaining label-association failures (dev15 + validation, v1c diagnostic)

55 wrongly associated labels: 30 whose target was never reconstructed (tick
marks, angle markers, figures outside the detected regions), 22 linked to the
wrong existing element (dimension labels inside unrelated circles or
rectangles, tick labels on the nearest segment), 3 unlinked. On `math_041`
four labels lie outside the only region found and are dropped before
association. These need geometry (V2) and leader/dimension evidence, not OCR.

## Not done in V1

- Grouping single characters into multi-character labels (MST grouping),
  orientation estimation and rotating slanted labels before OCR.
- Page-level text detection by Tesseract (it still only re-reads crops).
- Symbols: `x`/`×`, `-`/`—`, degree signs, units, coordinate pairs.
- Per-engine numbers are attributed by matching text and box origin; a label
  whose text or position changed after OCR shows as `unknown`.
- Lint: `ruff` is not installed here; `pyflakes` reports only the existing
  deliberate `import easyocr  # noqa: F401` availability check.
