# Accuracy optimisation, V2a: folded outlines, tick recovery, export checks

Same benchmark and protocol as `ACCURACY_V1.md`: `math100-v1` (synthetic,
exact ground truth), Model A only, Model B off, Tesseract on,
`OMP_NUM_THREADS=2`. Keep/revert decisions used `dev15`
(`evaluation/iterations/dev15.txt`) and validation only. The old 15-image test
split had already been inspected in V1, so it is now a **regression set**, not
a held-out set; its numbers below are not an accuracy claim. A fresh held-out
set for the final V1-vs-V2 claim is planned after V2b. n = 15 per split, so a
difference of one image is several points. Nothing here is evidence about real
tactile readers, and nothing here is a standards-compliance claim.

Tables: `python -m evaluation.report --split <split> --rescore <runs>`.

Runs:

- `v2base-*`: `main` at `1d2f08e` (V1 merged). It reproduces every accuracy
  number in `ACCURACY_V1.md`.
- `v2a5-*`: this PR.
- `v1d-test`: the V1 regression-set run, compared against `v2a5-test`.

Intermediate runs (`v2a1`..`v2a4`) were used for decisions and are not
committed. Two of them changed the design: `v2a3` showed lengthened ticks
running into braille that had been placed against the drawn tick, and `v2a4`
showed a closed figure (`math_063`'s octagon) removed by the density budget
and exported, which the evaluator counts as a critical problem.

## What changed

- **Strokes traced around both sides** (`vectorization.py`). A contour whose
  boundary pairs up with itself at stroke width and turns back at exactly two
  tips is one stroke traced twice. When the two tips almost meet it is a ring
  that broke in thresholding and is emitted once as a closed outline (collinear
  vertices dropped, so the break adds no corner); otherwise it is emitted as an
  open polyline, skipping segments already found as lines. Both carry
  `origin: "folded_outline"`. Filled shapes, junctions of several strokes and
  curves are left alone (synthetic tests, plus `math_055` circle and `math_063`
  octagon from the benchmark).
- **Tick marks** (`image_preprocessing.tick_ink_mask`, `vectorization`,
  `diagram_analysis`). Thin ticks were erased by the stroke filter before
  tracing. An unthinned dark-ink mask, relative to the bold stroke ink and off
  on dark paper, is used as evidence only for short strokes on a longer line.
  A tick is kept when it crosses the line, or when it is one of an evenly
  spaced series of similar one-sided strokes (`tick_provenance: "crossing"` /
  `"series"`). An isolated one-sided stroke is not a tick. Short lines are not
  suppressed globally.
- **Density budget** (`tactile_simplification.py`). Unconnected clutter goes
  before figure structure. Essential = sides joined by structural relationships,
  ticks, angle / right-angle / parallel markers, and closed figures at least
  5 mm across. If essential features still have to go, the action is
  `removed_density_essential` and QA blocks export
  (`density_removed_essential_geometry`). Short ticks are lengthened to the
  touch minimum instead of being dropped, and braille that a lengthened tick now
  runs into is moved to the nearest clear spot (`moved_braille_from_tick`);
  braille placement itself is the same search, now exposed as
  `clear_braille_position`.
- **Isolated specks**: a feature below touch size with no shared endpoint, no
  label and no marker or tick role is removed as `removed_isolated_speck`, with
  the reason in the simplification log. Small labelled outlines touching the
  figure are kept, so the minimum-size check can still block them.
- **Metrics** (`evaluation/`): "linked labels whose target was never built"
  and "export-blocking rate".

## Results

### dev15 (decisions)

| Metric | v2base | v2a5 |
|---|---:|---:|
| Object precision / recall (strict) | 0.440 / 0.589 | 0.469 / 0.670 |
| Object F1 (strict) | 0.504 | 0.551 |
| Essential object recall, final tactile | 0.633 | 0.696 |
| Object type accuracy | 0.742 | 0.773 |
| Label detection recall / precision | 0.540 / 0.865 | 0.540 / 0.865 |
| Label association accuracy | 0.647 | 0.676 |
| Linked labels whose target was never built | 19 | 13 |
| Relationship precision / recall / F1 | 0.540 / 0.308 / 0.392 | 0.553 / 0.400 / 0.464 |
| Edges represented once | 0.602 | 0.650 |
| Edge over-segmentation / duplicate / missed | 0.066 / 0.103 / 0.269 | 0.063 / 0.056 / 0.237 |
| Semantic preservation (end to end) | 0.376 | 0.413 |
| Simplification semantic preservation | 0.909 | 1.000 |
| Essential elements removed by simplification | 5 | 0 |
| QA pass rate / export-blocking rate | 0.800 / 0.200 | 0.867 / 0.133 |
| Critical problems exported | 0 | 0 |

Where it comes from: `math_022` (ticks found: strict object matches 2 -> 8,
correct label associations 2 -> 5), `math_044` and `math_063` (more objects,
relationships), `math_064` (the folded outline that caused 5 duplicates).
`math_063` changes from BLOCKED (42 of 82 features dropped) to exported with a
density warning: 26 of 66 unlabelled, non-essential features are dropped and
no essential ground-truth object is lost. It still loses labels (0 of 44
correct associations), so it is far from a good sheet.

### Validation (decisions)

| Metric | v2base | v2a5 |
|---|---:|---:|
| Object precision / recall (strict) | 0.258 / 0.539 | 0.255 / 0.539 |
| Object F1 (strict) | 0.349 | 0.346 |
| Essential object recall, final tactile | 0.831 | 0.831 |
| Object type accuracy | 0.604 | 0.625 |
| Label association accuracy | 0.778 | 0.778 |
| Linked labels whose target was never built | 11 | 11 |
| Relationship F1 | 0.358 | 0.358 |
| Edge over-segmentation / duplicate / missed | 0.117 / 0.017 / 0.216 | 0.117 / 0.017 / 0.216 |
| Semantic preservation (end to end) | 0.489 | 0.489 |
| QA pass rate / export-blocking rate | 0.867 / 0.133 | 0.867 / 0.133 |
| Critical problems exported | 0 | 0 |

**Got worse:** one false tick on `math_023` (a 38 px stroke on a line, kept as
a crossing tick), so precision 0.258 -> 0.255 and F1 0.349 -> 0.346.
**Not improved:** the ticks on `math_059` and `math_087` are still not found
(4 and 3 linked labels with no built target); the tick recovery generalised
from `math_022` on dev15 but not to these two.

### Old test split (regression set only)

| Metric | v1d-test | v2a5-test |
|---|---:|---:|
| Object F1 (strict) | 0.539 | 0.550 |
| Essential object recall, final tactile | 0.848 | 0.886 |
| Label association accuracy | 0.590 | 0.639 |
| Linked labels whose target was never built | 11 | 6 |
| Relationship F1 | 0.369 | 0.397 |
| Edges missed | 0.223 | 0.182 |
| Essential elements removed by simplification | 3 | 0 |
| QA pass rate / export-blocking rate | 0.933 / 0.067 | 0.933 / 0.067 |
| Critical problems exported | 1 | 0 |

`math_061` no longer exports with essential lines removed (3 -> 0 lost).
`math_076` is **still blocked** on `feature_below_minimum_size`, exactly as in
V1: the remaining small feature touches other geometry, so the isolated-speck
rule does not apply to it. It was not tuned further, because this split is not for
tuning. Object precision fell 0.448 -> 0.443 (`math_034`, `math_047` have more
predicted objects).

### Latency

Total P50 is higher in every V2a run (dev15 11.6 s -> 13.2 s, regression set
8.5 s -> 13.0 s), but per-stage timings put almost all of it in OCR, which this
PR does not change, so it is CPU contention between runs, not V2a. The stages
V2a touches cost, by median: preprocessing +25 to +35 ms and vectorisation
+40 to +60 ms per image. A clean one-run-at-a-time latency comparison is
**NOT MEASURED**.

## Limitations

- Tick recovery is tuned on axis and number-line ticks in this synthetic set;
  `math_059` / `math_087` show it misses other tick styles, and `math_023` shows
  one false positive.
- The folded-outline rule needs exactly two turn-back tips; a stroke folded
  through a junction is left as before.
- "Essential" is a rule (relationships, roles, markers, closed figures >= 5 mm),
  not ground truth. A figure feature with none of those can still be removed
  with only a warning.
- Moving braille away from a lengthened tick can fail when there is no clear
  spot nearby; QA then still blocks on `braille_on_line`.
- Angle and dimension targets, evidence-scored association and labels outside
  figure regions are V2b.
