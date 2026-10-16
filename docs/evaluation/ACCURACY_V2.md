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

## V2a follow-up: clean digital line art (thin strokes, degree signs, arcs, ticks)

Triggered by a teacher-supplied 480x360 JPEG of a quadrilateral with a
diagonal, two equal-length ticks, angle arcs labelled 70°, 25°, 35° and x, and
an `x = ?` caption. That image is **not** in the benchmark and nothing was
tuned on it alone; every rule below is checked on dev15 and validation.

On `main` (`d265a8d`) the image produced one bottom line, two outlines traced
around letters of the title, and the labels `25`, `70?`, `358`. The other three
sides, the diagonal, both ticks, every arc and both `x` labels were missing.

Cause: the noise check reads the crisp edges of clean digital art as noise, so
`enhance_copy` ran CLAHE plus sharpening. Threshold components went from 40
(2 smaller than 6 px) to 1,125 (1,047 smaller than 6 px); morphology then
broke the 1-2 px grey strokes into pieces, and OCR saw distorted glyphs.

### What changed

- **No sharpening for clean, full-contrast line art** (`image_enhancement`,
  `image_preprocessing.is_clean_line_art`): when the 1st-99th percentile range
  is at least 160 grey levels and almost no threshold components are specks,
  enhancement only normalises polarity.
- **Strokes that morphology shatters are kept whole** (`_shattered_strokes`):
  a long threshold component that opening breaks into 6 or more pieces is
  restored from the threshold mask.
- **Degree signs** (`ocr_postprocessing.restore_degrees`): `70"`, `708`, `70o`,
  `70?` or a bare `70` becomes `70°` only when a small raised glyph sits after
  the digits inside the OCR box, and that glyph is at least 2 px and 20% of the
  digit height (a 1 px speck after `13` stays `13`). It runs before text
  validation, which now allows `°`.
- **Every arc at a vertex** (`markers.detect_angle_arcs`): a vertex can carry
  several arcs (25° and 35° share one corner), and arcs clipped where the line
  mask cut their ends are bridged. Tick candidates where an arc crosses an arm
  are dropped (`drop_ticks_on_arcs`).
- **Short crossing ticks** (`vectorization._profile_ticks`): a tick no longer
  than the stroke is wide is found from the widened cross-section of its host
  line, but only when plain stroke continues on both sides, which rejects
  arrowheads (`math_046`, `math_096`).
- **Label association** (`label_association`): a degree measure prefers an
  angle whose arms enclose it near the arc radius, is never a point name, and a
  statement containing `=` is not linked to a tick.

### Supplied image, before and after

| | `main` | this branch |
|---|---|---|
| Sides + diagonal | 1 of 5 | 5 of 5 |
| Equal-length ticks | 0 of 2 | 2 of 2, one `equal_length` relationship |
| Labels read | `25`, `70?`, `358` | `70°`, `25°`, `35°`, `X`, `X=?` |
| Angles linked | none | 70° -> 69.9°, 25° -> 25.2°, 35° -> 36.4°, x -> 31.4° (top-left arc) |
| QA | passes (but only one line) | passes, score 86 |

Still wrong on that image: the title "Hard Geometry Problem ?" is not read,
`X=?` is linked to the right side rather
than to the angle x, and an unmarked 123.5° corner angle is emitted at the
top-right vertex with no arc.

### Benchmark (`v2a5` = `main`, `thin4` = this branch)

| Metric | dev15 v2a5 | dev15 thin4 | val v2a5 | val thin4 |
|---|---|---|---|---|
| Object precision (strict) | 0.469 | 0.458 | 0.255 | 0.286 |
| Object recall (strict) | 0.670 | 0.688 | 0.539 | 0.629 |
| Object F1 (strict) | 0.551 | 0.550 | 0.346 | 0.393 |
| Object type accuracy | 0.773 | 0.779 | 0.625 | 0.679 |
| Label detection recall | 0.540 | 0.540 | 0.627 | 0.636 |
| OCR CER (detected labels) | 0.212 | 0.206 | 0.211 | 0.166 |
| OCR WER (all essential labels) | 0.566 | 0.561 | 0.556 | 0.528 |
| Label association accuracy | 0.676 | 0.676 | 0.778 | 0.767 |
| Relationship F1 | 0.464 | 0.475 | 0.358 | 0.372 |
| Edges represented once | 0.650 | 0.661 | 0.667 | 0.719 |
| Edges missed | 0.237 | 0.220 | 0.216 | 0.176 |
| Edge duplicate rate | 0.056 | 0.055 | 0.017 | 0.024 |
| Quantitative labels exact | 0.347 | 0.360 | 0.587 | 0.635 |
| QA pass rate | 0.867 | 0.800 | 0.867 | 0.867 |
| Critical problems exported | 0 | 0 | 0 | 0 |

Regressions:

- **dev15 QA pass 0.867 -> 0.800 (`math_063`).** The degree check now reads
  a number-line `1` correctly instead of `1°`. As a point name it keeps its
  point under the 60-feature cap, which pushes out one essential line, so the
  export check blocks the page. The block is correct (exporting would drop
  figure geometry); the dense multi-figure page needs cropping either way.
- **Validation label association 0.778 -> 0.767** (`math_099`: 13 -> 12
  correct; `math_026` gained one). Object precision on dev15 fell
  0.469 -> 0.458 and the validation duplicate rate rose 0.017 -> 0.024, both
  from extra recovered strokes.

### Latency

Benchmark P50 rose (dev15 13.2 s -> 20.7 s), almost all in the OCR stage, but
these runs ran two at a time. Re-run one at a time on six dev15 images
(`math_001/002/004/014/044/063`), `main` took a median 25.0 s and this branch
18.5 s per image, so the machine's timing noise is larger than any difference
these changes make. A controlled latency claim either way is **NOT MEASURED**.

### Verification

Backend `pytest`: 776 passed, 10 skipped. Benchmark tests: 142 passed.
Frontend: 118 tests, typecheck and build pass. New regression tests are in
`backend/tests/test_line_art_marks.py`.
