# Baseline audit: what Model A does, and what the math benchmark can measure

Scope: the deterministic pipeline (`backend/app/services/pipeline.py`,
`build_full_analysis`) as merged on `main` after PRs #2-#7. This document is a
read-through of the code that the math benchmark exercises, written before any
benchmark numbers were produced, so the metric design follows the system
rather than the other way round.

## 1. Pipeline stages and what each one emits

| # | Stage (function) | Output used by the benchmark |
|---|---|---|
| 1 | `decode_image` | BGR array; failure = scored failed image |
| 2 | `assess_image_quality` | `QualityReport` (blur, contrast, resolution warnings) |
| 3 | `enhance_copy` | contrast-normalised copy; dark pages inverted |
| 4 | `preprocess_image` | binary ink mask: adaptive threshold, 3x3 open + erode, hairlines restored |
| 5 | `find_diagram_regions` / `mask_to_regions` | worksheet only: keeps figure regions, drops text blocks |
| 6 | `extract_shapes` | contours -> lines (Hough + merge), triangles, rectangles, polygons, circles, ellipses |
| 7 | `EasyOcrProvider.detect` + `postprocess_detections` | text boxes with strings and confidence |
| 8 | `drop_shapes_inside_text_regions`, `detect_right_angle_markers`, `drop_marker_strokes` | removes glyph strokes; right-angle markers become flags |
| 9 | `map_label_to_geometry`, `place_braille_markers` | label -> nearest geometry; Liblouis UEB braille; reading order (top-to-bottom, left-to-right) |
| 10 | `analyze_diagram` | `SemanticGeometry`: typed elements, points, angles, relationships |
| 11 | `simplify_geometry` | noise removal, collinear merges, contour simplification, 40-feature budget; every action logged |
| 12 | `run_tactile_qa` | mm-based checks on A4: stroke width, braille clearance, min feature, density |
| 13 | `render_tactile_svg` | final A4 SVG |

## 2. Model A vocabulary (what a prediction can be)

Emitted by the pipeline in practice: `line_segment`, `triangle`, `rectangle`,
`polygon`, `circle`, `ellipse`, `point`, `angle` (incl. right-angle markers),
`text_label`. Present in the enum but only reachable through teacher edits or
Model B: `ray`, `arrow`, `axes`, `arc`.

Consequences for scoring:

* A drawn ray, arrow, axis, number line or dimension line can at best be
  *found* as a `line_segment`; its type is then wrong. The benchmark reports
  geometric detection (family-compatible) and semantic classification
  (exact expected type) separately so this is visible rather than hidden.
* Curves (function graphs) have no type. They are scored for detection only and
  excluded from classification.
* Arcs and angle markers are usually broken into short lines or dropped.
* Model A has no diagram-level type, so diagram-type accuracy is
  `NOT MEASURED` for Model A.

## 3. Relationships

Model A infers `parallel_lines`, `perpendicular_lines`, `intersects`,
`connected_lines`, `endpoint_of`, `point_on_line`, `point_on_circle`,
`center_of` and angle associations between its *own* elements. Ground-truth
relations with no Model A counterpart (`VERTEX_OF`, `INSIDE`, `REFLECTION_OF`,
`TRANSLATION_OF`, `DIMENSION_OF`, `ANGLE_AT`) are counted and reported as
unsupported, not scored as misses. A ground-truth relation between an altitude
and a triangle *side* cannot be matched when Model A represents the triangle as
one closed contour; that is a representation limit, reported as a miss.

## 4. Known weak points before measuring (from earlier work in this repo)

* Low-resolution and half-scale uploads lose thin strokes (48-case STEM set:
  detection F1 0.51 on that variant).
* Pentagons and hexagons were not recognised as polygons in the STEM set.
* OCR misreads short labels and confuses I/l/1, O/0 (seen on real worksheets:
  "CLASS IX" -> "CLASS LX").
* Lines are fragmented at crossings and by dashes.
* Dense pages exceed the 40/60 tactile budget and are blocked by QA
  (intentional).

## 5. Why a new benchmark was needed

The existing `backend/benchmarks/stem/` set (48 cases, 12 figures x 4
variants) mixes biology, chemistry and physics, has bounding-box ground truth
only, and has no relationships, label roles, dimensions, essential/omittable
marking, or train/test split. It cannot answer "how accurate is TactileGeo on
school mathematics?". The new `evaluation/` benchmark keeps it untouched.

## 6. What the new benchmark captures per image

Everything in section 1, serialised by `evaluation/capture.py`: quality report,
raw shape count, OCR strings + boxes + braille + reading order, semantic and
simplified elements and relationships, every simplification action, QA issues,
the final SVG, stage timings (by wrapping the functions `pipeline.py` calls,
so the measured path is exactly `build_full_analysis`), plus run provenance
(git commit, dataset version, seed, model configuration, date).

## 7. Limits of the benchmark itself

* All 100 images are generated. Ground truth is exact by construction, but
  rendering (fonts, stroke styles, degradations) is narrower than real
  worksheets. Results are evidence about the *pipeline's behaviour on this
  distribution*, not a real-world accuracy figure. Real worksheets are a
  separate, verification-pending slice.
* No augmented variants of one parent exist, so there is no parent/child split
  leakage to control; every image is its own parent.
* Matching tolerances are fixed (section "Matching" in
  `evaluation/metrics.py`) and reported with every run.
