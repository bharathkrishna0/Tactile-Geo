# Multi-figure worksheet pages: one tactile sheet per figure

A worksheet page with several independent figures used to go through Model A as one
drawing. Header, Name/Date fields, answer boxes and watermark were traced, and every
figure shared one A4 feature budget, so density QA blocked export. This change splits
the page into figures (`backend/app/services/figure_split.py`, deterministic Model A
code) and processes each figure as its own tactile sheet.

## Data

`evaluation/generator/pages.py` builds `evaluation/dataset_pages` (`pages-v1`, seed
20261101): 30 pages of 1100×1556 px with 4, 6, 8 or 10 figures each, drawn with the
existing `Scene` templates, plus a header, Name/Date boxes, logo, dotted border, answer
boxes and a pale diagonal watermark. Conditions cycle clean digital / scanned /
photographed. Splits: dev 12, validation 9, test 9. The answer key records each
figure's box and which objects, points and labels belong to it.

`evaluation/run_pages.py --mode page` runs the whole page (the behaviour on `main`).
`--mode figures` splits the page, matches found figures to answer-key figures by box
IoU ≥ 0.3, runs Model A on each matched crop and scores it against that figure's part
of the answer key, moved into crop coordinates. An unmatched answer-key figure scores
as an empty output.

## Results (dev / validation, `pages-v1`)

Page mode = `main` behaviour (the whole-page path is unchanged; see the math100 check below).

| Metric | dev page | dev figures | val page | val figures |
|---|---|---|---|---|
| Figures found / answer key | 92 / 92 | 92 / 92 | 50 / 50 | 50 / 50 |
| Object F1 (strict, final) | 0.583 | **0.868** | 0.523 | **0.913** |
| Object precision | 0.537 | 0.910 | 0.409 | 0.935 |
| Essential object recall | 0.639 | 0.830 | 0.723 | 0.893 |
| Label detection recall | 0.559 | 0.835 | 0.573 | 0.818 |
| Label association accuracy | 0.655 | 0.779 | 0.791 | 0.854 |
| Relationship F1 (lower bound) | 0.509 | 0.654 | 0.383 | 0.648 |
| Relationship precision (lower bound) | 0.653 | **0.625** | 0.451 | 0.607 |
| Edges missed | 0.383 | 0.141 | 0.211 | 0.066 |
| Essential elements removed by simplification | 3 | 0 | 3 | 0 |
| Critical problems exported | 1 | 0 | 1 | 0 |
| Pages where every sheet passes QA | 0.83 | 1.00 | 0.89 | 0.89 |
| Figure sheets passing QA | – | 1.00 | – | 0.98 |

Worse: relationship precision on dev (0.653 → 0.625). Object matching for
`angle_marker` falls on dev (0.36 → 0.23), while the marker metric
`angle_markers.recall` rises (0.36 → 0.82). Object scoring counts an embossed
ordinary angle arc only as a generic `arc`, so the two disagree.

Segmentation on all 30 pages (dev+validation+test counts only, no Model A run):
198 answer-key figures, 200 found. The two extras are header text and a Name box on
one photographed page (`page_006`).

## math100 check (no splitting expected)

`build_full_analysis(figure=None)` only adds figure metadata, so the math100 results
should equal `main`'s. dev15 (`pages1-math-dev15` vs `dev15-trim2`, the `main` code):
all 9,839 numeric metrics identical except peak RSS. Validation (`pages1-math-validation` vs `val-trim2`): all 10,688 numeric metrics identical except peak RSS.

In the app, though, the API defaults a page with ≥ 2 figures to figure 1. On math100
the splitter returns several figures on 9 of 30 tuning/validation images
(`math_041`, `043`, `044`, `063`, `064`, `066`, `084`, `055`, `065`); most are real
multi-figure worksheets, but `math_084` (a table and a clip-art star) and `math_055`
(three polygons in one exercise) are split when they should not be. The teacher can
switch to "Whole page" in the figure picker.

## The supplied Class 7 Lines and Angles worksheet (not committed)

11 figures found: the 10 exercises and the cartoon. 10 of 11 sheets pass QA; figure 6
(`x =` / `y =` answer lines next to the triangle) is blocked by Braille on a line.
Before: one whole-page sheet, QA score 11, blocked. Still wrong: degree signs read as
digits (`600`, `1050`), blue arcs sometimes traced as extra circles, some point letters
missed. Those are the next PR.

## Limits

- Synthetic pages are drawn by the same templates as math100; real worksheets have
  more varied furniture. Clip art is not rejected.
- QA passing checks spacing and size, not mathematical correctness.
- Splitting adds about 0.05 s per image on this VM (`find_figures` timed alone);
  no other latency claim is made.
- The held-out `pages-v1` test split was not run.
