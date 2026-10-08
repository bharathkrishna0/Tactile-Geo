# Accuracy optimisation, iteration 5 (PR A): text-first OCR

Same benchmark and protocol as `ACCURACY_ITERATION_4.md`: `math100-v1`
(synthetic, exact ground truth), Model B off, tuning on `dev15` + validation
only, the 15-image test split run once after the candidate was chosen. Tables
come from `python -m evaluation.report`. n = 15 per split, so differences of a
few points are noise. The audit that motivated this work is
`ACCURACY_AUDIT.md`. Nothing here is evidence about real worksheets or real
tactile readers.

Runs:

- `base12-*`: the PR #12 code (`iter4d`), re-run on the same machine and under
  the same load as the candidates, so its latency is comparable. (`iter4d-*`
  latencies were recorded while other jobs shared the CPU, so they are about 2x
  higher; its accuracy numbers are identical to `base12`.)
- `iter5d-*` / `iter5g-test`: the selected configuration (Tesseract + OCR
  fusion, text read before tracing, glyph masking off). `iter5d` was produced
  with a scratch wrapper that disabled masking; it is equivalent to the
  committed default (`mask_text=False`).
- `iter5f-*`: the same plus glyph masking (`--mask-text-glyphs`), kept as the
  masking ablation.

## What changed

- **Second OCR engine (optional).** `tesseract_ocr.py` reads the isolated
  short-label crops (vertex letters, tick numbers) that EasyOCR's page detector
  misses. It runs only when `pytesseract` and the `tesseract` binary are
  present; otherwise the pipeline is EasyOCR-only, as before. Each call has a
  2 s timeout.
- **OCR fusion** (`ocr_fusion.py`). Both engines read the same crop; readings
  are never concatenated. Lookalikes (`O/0`, `I/l/1`, `S/5`, `B/8`, `Z/2`) are
  grouped, then the result is chosen by cross-engine agreement, per-engine
  confidence (EasyOCR alone >= 0.75, Tesseract alone >= 0.85, agreement >= 0.2)
  and whether the text is a plausible short maths label. Ambiguous
  letter/digit readings are resolved from the page (mostly letters vs mostly
  numbers). At most one label per crop. Every label carries `ocr_provider`
  (`easyocr`, `tesseract` or `easyocr+tesseract`) through the label mapping and
  benchmark captures.
- **Dashed-line suppression.** Runs of evenly spaced, same-size components
  (dashes) are not sent to OCR as glyphs.
- **Text before tracing.** `build_full_analysis` / `build_preview` now run OCR
  before `extract_shapes`. With masking off this does not change geometry.
- **Glyph-pixel masking (off by default).** `mask_text_glyphs` erases only ink
  components lying wholly inside a read text box, so a line that touches,
  crosses or leads into a label is kept whole. Stroke-shaped components (a
  `1`, `l`, `-`, or one dash of a dashed line read as `1`) are kept. Regression
  tests cover labels touching and beside lines, leader lines, dimension text
  between extension lines, angle labels inside angles and a dash read as a
  digit. It is available as `mask_text=True` and `--mask-text-glyphs`, but
  not enabled: see the decision below.

## Candidates (dev15 + validation only)

| Run | Change | Outcome |
|---|---|---|
| iter5a | Fusion + masking, EasyOCR solo >= 0.9, no Tesseract timeout | Best label recall (dev 0.63, val 0.70) but 3 images hit the 180 s limit, so not comparable or shippable |
| iter5b | + 2 s Tesseract timeout | No timeouts; label recall only +5 / +2 points; validation association 0.76 -> 0.62 |
| iter5c | + EasyOCR solo threshold back to 0.75 | Validation label recall 0.62; association still 0.62 |
| iter5d | iter5c with masking disabled | Label gains kept; association back to 0.75 on validation |
| iter5e | Masking with no box padding | Mixed; dropped |
| iter5f | Masking that keeps stroke-shaped components | Best dev15 (right-angle markers 0.86, association 0.68), but validation association 0.68 and relationship F1 0.32 vs 0.75 / 0.34 without masking |

Causes found while tuning: on a table image (`math_060`), erasing digits
changed how the grid lines were segmented and labels attached to different
lines (association 15 -> 10 correct). On `math_041`, EasyOCR read one dash of
a dashed altitude as `1`; erasing it broke the altitude and lost a right-angle
marker. The second case led to keeping stroke-shaped components.

### dev15 (n = 15)

| Metric (all, n=15) | base12-dev15 | iter5d-dev15 | iter5f-dev15 |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.416 | 0.416 | 0.451 |
| Object recall (strict) | 0.554 | 0.554 | 0.536 |
| Object F1 (strict) | 0.475 | 0.475 | 0.490 |
| Object F1 (structural: parts allowed) | 0.504 | 0.504 | 0.526 |
| Essential object recall, final tactile | 0.557 | 0.557 | 0.608 |
| Object type accuracy (matched) | 0.726 | 0.726 | 0.717 |
| Line endpoint error px, P50 | 1.494 | 1.494 | 1.180 |
| Line endpoint error px, P95 | 84 | 84 | 34 |
| Angle error deg, P50 | 0.182 | 0.182 | 0.177 |
| Polygon IoU, P50 | 0.989 | 0.989 | 0.989 |
| Circle centre error px, P50 | 12 | 12 | 12 |
| Point recall | 0.676 | 0.676 | 0.676 |
| Right-angle marker recall | 0.714 | 0.714 | 0.857 |
| Angle-arc marker recall | 1.000 | 1.000 | 1.000 |
| Label detection recall | 0.423 | 0.476 | 0.476 |
| Label detection precision | 0.827 | 0.851 | 0.851 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.648 | 0.611 | 0.611 |
| OCR CER (detected labels only) | 0.214 | 0.192 | 0.192 |
| OCR WER (all essential labels) | 0.656 | 0.604 | 0.604 |
| Label association accuracy | 0.625 | 0.644 | 0.678 |
| Braille end-to-end exact | 0.762 | 0.811 | 0.811 |
| Braille cell error rate | 0.179 | 0.159 | 0.159 |
| Braille reading-order concordance | 0.920 | 0.930 | 0.936 |
| Relationship recall (supported, final) | 0.243 | 0.265 | 0.270 |
| Relationship precision (mappable predictions) | 0.469 | 0.490 | 0.459 |
| Relationship F1 | 0.320 | 0.344 | 0.340 |
| Edges represented once (correct grouping) | 0.521 | 0.521 | 0.516 |
| Edge over-segmentation rate | 0.078 | 0.078 | 0.079 |
| Edge duplicate rate | 0.069 | 0.069 | 0.070 |
| Edges missed | 0.376 | 0.376 | 0.387 |
| Semantic Preservation Rate (end to end) | 0.298 | 0.332 | 0.344 |
| Simplification semantic preservation | 0.898 | 0.898 | 0.980 |
| Essential elements removed by simplification | 5 | 5 | 1 |
| Quantitative labels exact | 0.320 | 0.333 | 0.333 |
| Tactile QA pass rate | 0.800 | 0.800 | 0.867 |
| Critical problems exported (not blocked) | 0 | 0 | 1 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 4,293 | 4,328 | 4,602 |
| Total latency ms, P95 | 5,257 | 10,822 | 10,822 |

### Validation (n = 15)

| Metric (all, n=15) | base12-validation | iter5d-validation | iter5f-validation |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.262 | 0.262 | 0.269 |
| Object recall (strict) | 0.539 | 0.539 | 0.517 |
| Object F1 (strict) | 0.353 | 0.353 | 0.354 |
| Object F1 (structural: parts allowed) | 0.406 | 0.406 | 0.421 |
| Essential object recall, final tactile | 0.831 | 0.831 | 0.831 |
| Object type accuracy (matched) | 0.604 | 0.604 | 0.587 |
| Line endpoint error px, P50 | 1.267 | 1.267 | 1.383 |
| Line endpoint error px, P95 | 39 | 39 | 25 |
| Angle error deg, P50 | 0.069 | 0.069 | 0.080 |
| Polygon IoU, P50 | 0.986 | 0.986 | 0.987 |
| Circle centre error px, P50 | n/a | n/a | n/a |
| Point recall | 0.750 | 0.750 | 0.750 |
| Right-angle marker recall | 0.667 | 0.667 | 0.667 |
| Angle-arc marker recall | 1.000 | 1.000 | 1.000 |
| Label detection recall | 0.576 | 0.619 | 0.619 |
| Label detection precision | 0.946 | 0.915 | 0.915 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.461 | 0.441 | 0.441 |
| OCR CER (detected labels only) | 0.219 | 0.213 | 0.213 |
| OCR WER (all essential labels) | 0.597 | 0.562 | 0.562 |
| Label association accuracy | 0.758 | 0.747 | 0.676 |
| Braille end-to-end exact | 0.727 | 0.747 | 0.747 |
| Braille cell error rate | 0.244 | 0.236 | 0.236 |
| Braille reading-order concordance | 0.871 | 0.868 | 0.865 |
| Relationship recall (supported, final) | 0.337 | 0.337 | 0.326 |
| Relationship precision (mappable predictions) | 0.345 | 0.345 | 0.315 |
| Relationship F1 | 0.341 | 0.341 | 0.320 |
| Edges represented once (correct grouping) | 0.660 | 0.660 | 0.673 |
| Edge over-segmentation rate | 0.118 | 0.118 | 0.099 |
| Edge duplicate rate | 0.017 | 0.017 | 0.025 |
| Edges missed | 0.222 | 0.222 | 0.209 |
| Semantic Preservation Rate (end to end) | 0.454 | 0.474 | 0.459 |
| Simplification semantic preservation | 1.021 | 1.021 | 1.021 |
| Essential elements removed by simplification | 0 | 0 | 0 |
| Quantitative labels exact | 0.556 | 0.587 | 0.587 |
| Tactile QA pass rate | 0.933 | 0.867 | 0.800 |
| Critical problems exported (not blocked) | 0 | 0 | 0 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 4,307 | 4,622 | 4,631 |
| Total latency ms, P95 | 6,203 | 10,468 | 9,090 |

Text-induced geometry (scratch diagnostic: predicted elements whose box lies
inside a ground-truth label box), totals over 15 images: dev15 base12 17,
iter5d 17, iter5f 17; validation base12 41, iter5d 42, iter5f 41. Masking did
not reduce it on these images.

## Decision

Keep Tesseract + fusion; keep glyph masking **off**.

- Label recall rose on both tuning splits (dev 0.423 -> 0.476, validation
  0.576 -> 0.619), CER and WER fell, label precision stayed above 0.85,
  association held (dev 0.625 -> 0.644, validation 0.758 -> 0.747), and
  relationship F1 and object recall did not regress.
- Masking passed its unit tests but did not meet the retention rule: it did not
  reduce text-induced geometry, and on validation it lowered association,
  relationship F1 and QA pass rate. Its gains on dev15 did not carry over.

## Held-out test (n = 15, run once)

| Metric (all, n=15) | base12-test | iter4d-test | iter5g-test |
|---|---|---|---|
| Images failed (error/timeout) | 0 | 0 | 0 |
| Object precision (strict) | 0.445 | 0.445 | 0.448 |
| Object recall (strict) | 0.676 | 0.676 | 0.676 |
| Object F1 (strict) | 0.537 | 0.537 | 0.539 |
| Object F1 (structural: parts allowed) | 0.592 | 0.592 | 0.594 |
| Essential object recall, final tactile | 0.835 | 0.835 | 0.835 |
| Object type accuracy (matched) | 0.667 | 0.667 | 0.667 |
| Line endpoint error px, P50 | 1.368 | 1.368 | 1.368 |
| Line endpoint error px, P95 | 39 | 39 | 39 |
| Angle error deg, P50 | 0.181 | 0.181 | 0.181 |
| Polygon IoU, P50 | 0.990 | 0.990 | 0.990 |
| Circle centre error px, P50 | 0.000 | 0.000 | 0.000 |
| Point recall | 0.889 | 0.889 | 0.889 |
| Right-angle marker recall | 0.667 | 0.667 | 0.667 |
| Angle-arc marker recall | 0.000 | 0.000 | 0.000 |
| Label detection recall | 0.399 | 0.399 | 0.474 |
| Label detection precision | 0.866 | 0.866 | 0.863 |
| OCR CER (all essential labels; missed label = all chars wrong) | 0.692 | 0.692 | 0.664 |
| OCR CER (detected labels only) | 0.369 | 0.369 | 0.358 |
| OCR WER (all essential labels) | 0.759 | 0.759 | 0.712 |
| Label association accuracy | 0.600 | 0.600 | 0.567 |
| Braille end-to-end exact | 0.583 | 0.583 | 0.621 |
| Braille cell error rate | 0.274 | 0.274 | 0.263 |
| Braille reading-order concordance | 0.979 | 0.979 | 0.947 |
| Relationship recall (supported, final) | 0.321 | 0.321 | 0.321 |
| Relationship precision (mappable predictions) | 0.406 | 0.406 | 0.409 |
| Relationship F1 | 0.358 | 0.358 | 0.360 |
| Edges represented once (correct grouping) | 0.710 | 0.710 | 0.710 |
| Edge over-segmentation rate | 0.035 | 0.035 | 0.035 |
| Edge duplicate rate | 0.052 | 0.052 | 0.052 |
| Edges missed | 0.223 | 0.223 | 0.223 |
| Semantic Preservation Rate (end to end) | 0.339 | 0.339 | 0.363 |
| Simplification semantic preservation | 0.943 | 0.943 | 0.943 |
| Essential elements removed by simplification | 4 | 4 | 4 |
| Quantitative labels exact | 0.230 | 0.230 | 0.279 |
| Tactile QA pass rate | 1.000 | 1.000 | 1.000 |
| Critical problems exported (not blocked) | 1 | 1 | 1 |
| Images over 60 features | 0 | 0 | 0 |
| Total latency ms, P50 | 5,151 | 11,016 | 5,261 |
| Total latency ms, P95 | 5,555 | 11,895 | 12,961 |

Text-induced geometry: base12 35, iter5g 34.

What the test run shows:

- **Better:** label recall 0.399 -> 0.474, CER 0.692 -> 0.664, WER 0.759 ->
  0.712, Braille end-to-end exact 0.583 -> 0.621, quantitative labels exact
  0.230 -> 0.279, semantic preservation 0.339 -> 0.363. Geometry, markers and
  relationships are unchanged (as expected, since masking is off).
- **Worse:** label association accuracy 0.600 -> 0.567. The number of
  correctly associated labels rose (30 -> 34), but the 10 newly found labels
  were associated correctly only 4 times, so the rate fell. Label association
  with leader-line and dimension evidence is PR B.
- **Latency:** P50 5.2 s -> 5.3 s, P95 5.6 s -> 13.0 s. Images with many glyph
  candidates make up to 80 Tesseract calls (about 65 ms each).
- **Unchanged problem:** 1 test sheet still exports with a critical problem
  (an essential element removed by simplification), as on PR #12. PR B.

## Not done in this PR

The approved plan had more than this. Not built yet:

- A dedicated text-candidate stage with connected-component classification,
  MST grouping of characters into multi-character labels, orientation
  estimation, and rotating angled labels to horizontal before OCR (with
  coordinates mapped back). Glyph candidates still come from the existing
  isolated-glyph finder (iteration 3).
- Fusion of overlapping page-level detections from both engines. Tesseract
  only re-reads crops; it does not detect text on the page.
- Handling of `x`/`×`, `-`/`—`, degree signs, units and coordinate pairs.
- Per-engine metrics in the benchmark report. The per-label `ocr_provider` is
  captured, but there is no EasyOCR-only vs Tesseract-only table.
- A formal text-induced-geometry metric (the numbers above come from a scratch
  script).
- Docker image size with Tesseract was not measured.
