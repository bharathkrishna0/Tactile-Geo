# Accuracy report: TactileGeo on math100-v1

These are machine metrics on **100 generated mathematics images** whose
ground truth is exact because it is recorded while drawing. They describe
how the pipeline behaves on this distribution. They are **not** a real-world
accuracy figure, not evidence about learning outcomes (no participant study
has been run, see `HUMAN_EVALUATION_PROTOCOL.md`), and not a claim of
compliance with any tactile-graphics standard. Real NCERT pages are reported
separately in section 7, without accuracy.

## 1. Runs used

| Run | Images | What it is |
|---|---|---|
| `model_a_v2-all` | 100 (all splits) | Model A at commit 5733d2d, 180 s per-image limit (1 timeout: math_076) |
| `model_a_v2-test-t600` | 15 (test) | Same Model A, 600 s limit, 0 failures. **Baseline for every test-split comparison** |
| `model_a_v2+model_b-none-test` | 15 | Model B called, teacher accepts nothing |
| `model_a_v2+model_b-accept_all-test-relfix` | 15 | Simulated teacher accepts every finding with a Model A match |
| `model_a_v2+model_b-oracle-test-relfix` | 15 | Simulated teacher accepts only findings whose kind matches ground truth |
| `model_a_v2+unet-test` | 15 | Model A binary ∩ U-Net mask, see `UNET_FEASIBILITY.md` |
| `unet_only-test` | **11 of 15** | U-Net mask replaces binarisation; stopped on request |
| `real_ncert` | **13 of 20** | NCERT pages, no ground truth; stopped on request |

Superseded runs (`model_a-all`, the pre-`relfix` Model B runs) are git-ignored.
`relfix` means after the fix in `geometry_relations._is_line_like` that keeps
line relationships for elements a teacher retyped to `axes`/`arrow`.

Dataset: `math100-v1`, seed 20261002, 20 images per tier (easy, moderate,
hard, very_hard, adversarial), categories A–L, split 70/15/15. The test split
has 3 images per tier. Matching: optimal one-to-one assignment, an object
matches when its symmetric mean outline distance is within
`max(4 px, 1.5% of the image diagonal)`. "Strict" requires one predicted
object per ground-truth object; "structural" also credits a shape found as its
parts (e.g. a triangle found as 3 segments).

## 2. Headline: Model A, all 100 images (`model_a_v2-all`)

| Metric | Value |
|---|---:|
| Images failed | 1 (timeout at 180 s; completes in 209 s with 600 s) |
| Object F1, semantic stage (strict / structural) | 0.367 / 0.410 |
| Object F1, final tactile (strict / structural) | 0.400 / 0.445 |
| Essential object recall, final tactile (structural) | 0.693 |
| Object type accuracy (matched objects) | 0.580 |
| Point recall / localisation P50 | 0.709 / 1.95 px |
| Line endpoint error P50 / P95 | 1.48 / 73.2 px |
| Angle error P50 / P95 | 0.13° / 0.86° |
| Polygon IoU P50 | 0.989 |
| Circle centre error P50 / radius error P50 | 2.24 px / 1.8% (n=8 matched) |
| Right-angle marker recall | **0.000** (0 of 24) |
| Label detection recall / precision | **0.326** / 0.959 |
| OCR CER, all essential labels / detected only | 0.671 / 0.238 |
| Exact label text when detected | 0.728 |
| Label association accuracy | 0.289 |
| Braille end-to-end exact (labels evaluated) | 0.729 |
| Braille cell error rate | 0.218 |
| Braille collisions / Braille-on-line | 0 / 4 |
| Relationship recall (supported types, final) | 0.135 |
| Simplification semantic preservation | 0.979 (8 essential elements removed) |
| Quantitative labels preserved exactly | 0.290 |
| Tactile QA pass rate | 0.900 |
| SVG present / valid | 0.99 / 1.00; 0 out-of-bounds, 0 duplicates, 10 self-intersecting polygons |
| Embossed features P50 / P95 | 7 / 40 (none over 60) |
| Total latency P50 / P95 | 12.9 s / 55.3 s (OCR is 98% of it) |
| Peak memory | 3.6 GB RSS |

### By difficulty (all 100)

| Tier | n | Obj F1 strict | Obj F1 struct | Essential final | Label recall | CER | Rel recall | QA pass | Failed |
|---|---|---|---|---|---|---|---|---|---|
| easy | 20 | 0.527 | 0.561 | 0.841 | 0.636 | 0.341 | 0.169 | 0.900 | 0 |
| moderate | 20 | 0.406 | 0.461 | 0.810 | 0.595 | 0.409 | 0.167 | 0.850 | 0 |
| hard | 20 | 0.363 | 0.409 | 0.733 | 0.207 | 0.724 | 0.127 | 0.900 | 0 |
| very_hard | 20 | 0.361 | 0.401 | 0.550 | 0.080 | 0.889 | 0.104 | 0.950 | 1 |
| adversarial | 20 | 0.291 | 0.329 | 0.790 | 0.470 | 0.566 | 0.147 | 0.900 | 0 |

Accuracy falls with difficulty as designed. `very_hard` (dense mixed
worksheets) is the worst tier for labels and essential geometry.
`adversarial` has the lowest object F1 because of distractors, noise and
near-miss shapes, but its labels survive better than `hard`/`very_hard`.

### By category (final tactile object F1, strict)

| Category | n | F1 |
|---|---:|---:|
| E measurement | 7 | 0.927 |
| G angles | 9 | 0.885 |
| F shapes/polygons | 8 | 0.688 |
| A geometry figure | 9 | 0.667 |
| B worksheet geometry | 11 | 0.553 |
| H circles | 10 | 0.543 |
| L mixed worksheet | 9 | 0.405 |
| D coordinate geometry | 9 | 0.352 |
| C graph | 8 | 0.246 |
| K tables/diagrams | 8 | 0.215 |
| J fractions/number lines | 6 | 0.190 |
| I transformations | 6 | 0.162 |

Line-based figures (measurement, angles) are handled well. Graphs, tables,
number lines and transformations are poor because Model A has no `bar`,
`tick`, `curve`, `axes` or `arrow` output type (axes become `line_segment` in
52 of 52 matched cases), and because their many short strokes fragment.

### Per ground-truth type (recall, strict)

axis 0.96, dimension line 1.00, number line 1.00, arrow 1.00, ray 0.88,
line segment 0.71, rectangle 0.71, polygon 0.50, circle 0.47, table line
0.32, angle marker 0.25, bar 0.00, curve 0.00 (bars/curves 0.53/0.50
structural, i.e. found only as parts).

## 3. What the numbers mean

1. **Where geometry is found, it is accurate.** Median endpoint error 1.5 px,
   angle error 0.13°, polygon IoU 0.99. The P95 endpoint error (73 px) comes
   from lines that are split or merged, not from imprecise ones.
2. **Missing labels are the main weakness.** Only a third of essential labels
   are found. A found label is usually right (precision 0.96, 73% exact), so
   the all-labels CER of 0.67 is mostly labels never found, where every
   character counts as wrong.
3. **OCR CER definitions.** "All essential labels" counts a missed label as
   fully wrong. "Detected only" scores only labels that were found. On the
   test split these are 0.720 and 0.342.
4. **Relationship recall is low (0.135),** but the reasons differ by type.
   PARALLEL (0.73) and PERPENDICULAR (0.65) recall is fine, though Model A
   predicts many more of them than exist, so the precision lower bound is
   0.02. ON recall is 0 (0 of 295). About 60% of those are ticks/bars on an
   axis, which Model A cannot output. For the rest, Model A emits
   `point_on_line` for segment endpoints that the ground truth calls
   ENDPOINT_OF. CENTER_OF is 0 because Model A has no centre-point output.
   VERTEX_OF, LABELS, INSIDE, DIMENSION_OF, ANGLE_AT, REFLECTION_OF and
   TRANSLATION_OF have no Model A equivalent and are reported as unsupported,
   not as misses.
5. **Simplification rarely hurts.** 98% of essential elements found before
   simplification are still present after it.
6. **QA blocks are real problems.** The 9 QA blocks are
   `feature_below_minimum_size` (5) and `braille_on_line` (4); the tenth
   non-passing image is the timeout. No sheet exceeds the 60-feature hard limit.

## 4. Model B and teacher ablation (test split, 15 images)

Free model only: `dots-studio/dots-3-note-preview:free` via OpenRouter, 150 s
timeout, up to 3 attempts, cost $0. Model B sees the image, not Model A's
output. Accepted findings only change Model A elements, through
`semantic_v2.py`, and never coordinates.

| Metric | Model A | + Model B, accept none | + accept all | + oracle teacher |
|---|---:|---:|---:|---:|
| Object F1 strict / structural | 0.519 / 0.572 | 0.519 / 0.572 | 0.519 / 0.572 | 0.519 / 0.572 |
| Essential recall, final | 0.861 | 0.861 | 0.861 | 0.861 |
| **Object type accuracy** | 0.652 | 0.652 | **0.681** | **0.681** |
| Label recall / CER | 0.263 / 0.720 | same | same | same |
| Relationship recall | 0.109 | 0.109 | 0.109 | 0.109 |
| Tactile QA pass rate | 1.000 | 1.000 | 1.000 | 1.000 |
| Total latency P50 | 37 s | 104 s | 116 s | 112 s |

Provider outcome (same cached responses in all three Model B runs):

| | |
|---|---|
| Usable responses | **5 of 15** (33%) |
| `invalid_response` | 10 (7 empty responses, 3 cut-off JSON) |
| `rate_limited` / `timeout` / `api_error` | 0 / 0 / 0 |
| Provider latency P50 | 80 s |
| Findings accepted (accept_all / oracle) | 9 / 6 |
| Changes applied (both policies) | 4 confirmations, 2 type changes |
| Kind correct where paired with ground truth | Model B 6/7 (0.86) vs Model A 5/7 (0.71) |

Interpretation:

* By design, Model B can only fix types and confirmations. It cannot add a
  missed shape or label, so detection, OCR and geometry are identical across
  the four columns.
* Its one measurable effect is type accuracy, 0.652 → 0.681, from 2 retyped
  elements. Accept-all and oracle produce the same output because the 3
  findings only accept-all took had no Model A element to apply to.
* On this provider, two in three calls produced nothing usable, and a usable
  answer took about 80 s. Failed calls count as "no help", never as correct
  predictions.
* 15 images and 5 usable answers are too few to generalise. This is a test
  of one free model on one day, not a judgement on vision-language models.

## 5. U-Net mask (test split)

Summary only; details and the decision (**KEEP AS EXPERIMENTAL**) are in
`UNET_FEASIBILITY.md`. Model A + U-Net raised strict object F1 from 0.519
to 0.638 and essential recall from 0.861 to 0.911. It lowered QA pass from
1.000 to 0.933 and did not change labels or OCR. Synthetic data only.

## 6. Latency

| Stage (all 100, P50 / P95) | ms |
|---|---:|
| OCR (EasyOCR, CPU) | 12,645 / 54,305 |
| Preprocessing | 80 / 231 |
| Vectorisation | 10 / 47 |
| Semantic analysis | 0.5 / 10 |
| Braille + label layout | 4 / 44 |
| Simplification, QA, SVG | < 10 each |
| **Total** | **12,906 / 55,276** |

Test-split latencies in sections 4–5 were measured while other runs shared
the CPU, so they are higher than the all-images figure and differences
between them are not clean.

## 7. Real slice: NCERT pages (no ground truth)

`python -m evaluation.run_real --slice ncert`, summary
`evaluation/results/real_ncert/summary.json`. **13 of 20 pages processed
before the run was stopped on request.** Accuracy: **NOT MEASURED (no
verified annotations).**

| Measure (13 pages) | Value |
|---|---|
| Pipeline failures | 0 |
| QA pass | 8 of 13 (0.615) |
| Blocking reasons | feature_below_minimum_size 4, braille_on_line 2 |
| Embossed features P50 / P95 | 36 / 58; 5 pages over the 40 target, 0 over the 60 limit |
| OCR labels found P50 | 10 per page |
| SVG valid | 13 of 13 |
| Latency P50 / P95 | **288 s / 487 s** per page |

Dense textbook pages make EasyOCR the bottleneck: about 5 minutes per page
on this CPU, about 20 times the synthetic median. This number matters most for a
classroom: a teacher would wait minutes per page. Side-by-side outputs are
written to `evaluation/reports/examples/real_ncert/` locally and are
git-ignored, because the pages may not be republished.

## 8. Limitations

* The synthetic images come from one generator. Real fonts, handwriting,
  textbook layouts and photo conditions are wider than this.
* The test split has 15 images (3 per tier). Differences of a few points
  between runs are within noise; no confidence intervals are claimed.
* Model B results reflect one free model and 5 usable answers.
* Some runs were stopped early: `unet_only-test` (11/15) and `real_ncert` (13/20).
* Relationship precision is a lower bound, because Model A predictions that
  map to no ground-truth object are counted as wrong.
* Diagram-type accuracy is NOT MEASURED; Model A produces no diagram-level
  type.

## 9. Reproduce

```bash
python -m evaluation.generator.build               # rebuild math100-v1 (deterministic)
python -m evaluation.run_benchmark --split all --model-a --run-id model_a_v2-all
python -m evaluation.run_benchmark --split test --model-a --run-id model_a_v2-test-t600 --image-timeout 600
MODEL_B_ENABLED=true OPENROUTER_API_KEY=... OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free MODEL_B_TIMEOUT_S=150 \
  python -m evaluation.run_benchmark --split test --model-a --model-b --teacher accept_all
python -m evaluation.report model_a_v2-test-t600 model_a_v2+model_b-accept_all-test-relfix model_a_v2+unet-test --split test
python -m evaluation.failure_analysis model_a_v2-all --model-b-run model_a_v2+model_b-accept_all-test-relfix --unet-run model_a_v2+unet-test
python -m evaluation.run_real --slice ncert
```
