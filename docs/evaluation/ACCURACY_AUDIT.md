# Accuracy audit: current pipeline against the text-first architecture

This audit compares the code on the PR #12 branch (`iter4d`) with the target
pipeline (image quality -> text-first analysis -> two OCR engines -> OCR fusion ->
label grouping/association -> Model A geometry -> topology -> optional Model B ->
fusion -> semantic graph -> tactile simplification -> textures + Braille ->
constraint layout -> QA gate -> teacher review -> SVG). It complements
`BASELINE_AUDIT.md` (the original pre-optimisation audit) rather than replacing it.

Evidence rules: measurements use the synthetic `math100-v1` benchmark, dev15 +
validation only (35 images, 304 essential labels). The held-out test split is run
once per PR, after the candidate is chosen. Real NCERT worksheets have no verified
ground truth and are not scored here. Nothing below is a claim of standards
compliance or of tactile accuracy for real readers.

What each later PR changes is recorded in its iteration report
(`ACCURACY_ITERATION_5.md` for the text-first OCR work).

## 1. Where the code stood against the target pipeline

| Stage in the target pipeline | Current code | Gap |
|---|---|---|
| Image quality gate | `image_quality.py`, `image_enhancement.py` (report + enhanced working copy) | Report does not change routing; no "reject / ask for rescan" path |
| Text-first analysis | **Not text-first.** `pipeline.py` runs `extract_shapes` first, OCR second, then deletes shapes inside OCR boxes | Biggest structural gap |
| Text detection | EasyOCR's own detector on the whole page, plus a single-glyph re-read (`ocr.py`) | No text-candidate stage independent of the OCR engine |
| EasyOCR + second engine | EasyOCR only, behind `OcrProvider` protocol | No second engine. Tesseract 4.1.1 installs from apt; PaddleOCR not tried |
| OCR fusion | None (one engine) | Missing |
| Label grouping | EasyOCR line grouping only | No baseline/spacing grouping, no rotated text |
| Label association | `label_association.py`: nearest point, text continuation, edge proximity, with reasons | No leader-line / arrow / dimension-line evidence |
| Model A primitives + reconstruction | `vectorization.py`, `segment_geometry.py` (collinear merge, closed shapes) | Works; edge over-segmentation 3.5%, missed edges 22% |
| Topology / relationships | `geometry_relations.py`, markers in `markers.py`, `right_angles.py` | Pairwise with filtering; relations carry evidence but no `CONNECTED_TO` / `ANGLE_BETWEEN`, no uniform `source` field |
| Model B, Fusion | `model_b/` advisory only; fusion pairs findings to Model A elements; teacher apply via `semantic_v2.py` | Model B does not see Model A ids; no structured `ASSOCIATE_LABEL`-style actions; no explicit `UNCERTAIN` result for unmatched claims |
| Semantic geometry graph | `SemanticGeometry` (elements + relationships) | Good base; needs provenance states (OCR detected / AI inferred / teacher corrected) |
| Tactile simplification | `tactile_simplification.py`: noise removal + 40-feature budget, every drop logged with a reason | Budget can drop essential lines (math_061 lost 4 in iter4d) |
| Texture rules | **Missing.** SVG is stroke-only, `fill:none` | No semantic fills, no legend |
| Braille rules | Liblouis UEB Grade 2 (`braille.py`), separate from layout | OK |
| Constraint-based layout | `braille_layout.py`: fixed ring of candidate offsets, first fit wins | Not a scored search; no density or distance-to-feature term |
| QA gate PASS / REVIEW | `tactile_qa.py`: severities + readiness score; `passes` = no errors | No explicit PASS/WARNING/BLOCKED object; 1 test sheet exports with a critical problem |
| Teacher correction → recompile | Editor + Model B apply both re-run simplification → layout → QA → SVG | Exists |

## 2. New measurement: why labels are missed

Label recall is 0.40 on test, and labels drive association, Braille and the semantic
preservation score. I split the 304 dev15+validation labels by condition:

| Condition | Label recall |
|---|---|
| clean digital | 0.88 |
| scanned | 0.66 |
| skew | 0.30 |
| low resolution | 0.33 |
| low contrast | 0.21 |
| photographed | 0.17 |
| blur | 0.00 |

Also: single-character labels 0.44, labels under 16 px tall 0.16, vertex labels 0.38,
tick labels 0.31.

**Detection or reading?** I cropped every label at its ground-truth box and read the crop
with each engine (an oracle test of reading only):

| Labels | n | Tesseract reads exactly | EasyOCR reads exactly | Either engine |
|---|---|---|---|---|
| Found by the pipeline today | 148 | 0.88 | 0.82 | 0.93 |
| **Missed by the pipeline today** | 156 | 0.55 | 0.54 | **0.69** |

So most missed labels are **not found**, rather than misread: given the right box,
the two engines together read 69% of them. Finding text regions first is the main lever,
and a second engine adds about 15 points over either one alone on hard crops.
Tesseract cost about 70 ms per crop on this CPU (EasyOCR recognition about 11 ms).
This is an upper bound: a real detector will not find every box.

Most common misreads on found labels: `O`→`0` (7 times), `°`→`0`/`o`, units (`4 cm`→`4`,
`7 cm`→`Tem`), coordinate pairs (`D(5, 4)`→`DG,4)`).

## 3. Points from the UW paper used in the plan

The paper (ASSETS '07) is a workflow reference, not evidence about TactileGeo.
Three ideas map directly to our gaps:
- **Find characters before reading them:** classify connected components as text or not
  using height, width, area and radial density (they used an SVM). We can do this with
  rules tuned on dev data.
- **Group characters into labels** with a minimum spanning tree over character centroids,
  using line-fit angle, fit error and spacing. This also gives the rotation angle
  (perpendicular least-squares fit) for rotating a label horizontal before OCR.
- **Label placement as scored search:** score = a·(overlap with other labels) +
  b·(overlap with drawing) + c·(distance moved), with labels moved greedily from a
  priority queue. Ours currently takes the first free offset.

## 4. Order of work

- **PR A (text-first OCR):** second OCR engine (Tesseract) on label crops, OCR
  fusion with math plausibility, OCR before tracing, glyph-pixel masking.
  See `ACCURACY_ITERATION_5.md` for what was built, what was measured, and what
  was deferred.
- **PR B:** simplification must not drop essential geometry; explicit
  PASS / WARNING / BLOCKED gate where any critical issue blocks; leader-line and
  dimension-line evidence for label association.
- **PR C:** relationship graph with uniform evidence/source fields and
  provenance states; scored Braille placement; texture rules with a legend.
- **PR D:** Model B sees Model A ids and returns structured, teacher-mediated
  actions; unsupported claims become `UNCERTAIN`, never geometry.
