# Edutactile-inspired optimization: Phase 0 audit

Status: audit only, no production code changed. Based on reading `main` (after PRs #16–#18).
The Edutactile paper itself was not available when this was written; the
"Edutactile idea" column follows the task brief's description of it and must be
re-checked against the paper.

## Baseline already measured on this code

`main` contains the text-first OCR code measured as iteration 5 (`ACCURACY_ITERATION_5.md`).
The most recent runs on that code (math100-v1, 15 images per split):

| Metric | dev15 (`iter5f`) | validation (`iter5f`) |
|---|---:|---:|
| Object F1 (strict) | 0.490 | 0.354 |
| Object F1 (structural) | 0.526 | 0.421 |
| Essential object recall, final tactile | 0.608 | 0.831 |
| Relationship F1 | 0.340 | 0.320 |

Iteration 0 will re-run dev15 + validation on `main` before any change and add the metrics
that do not exist yet: duplicate geometry rate, line fragmentation rate, geometric
displacement, Braille collisions, critical QA failures, region reconstruction and
texture conflicts. Until then these are **NOT MEASURED**. The held-out test split is run
once, at the end.

## Feature-by-feature audit

### 1. Structured geometry representation
- **Current:** `DetectedElement` (`models/geometry.py`) already has a stable `id` (`el_N`), `type`,
  `geometry`, `bbox`, `confidence`, `confidence_level`, `needs_review`, `source`, `provenance`,
  `semantic_properties` and `associated_label_id`. `ElementRelationship` holds typed relations
  (vertex_of, endpoint_of, parallel, equal_length, label_object, ...). Edits go through
  `services/editing.py`, and `TransformationExplanation` records stage + message.
- **Missing:** explicit `semantic_importance` (it is computed implicitly by
  `tactile_simplification.is_protected`), `semantic_role`, and per-element tactile rendering
  properties. Output (page mm) coordinates are not stored; the SVG reuses source px with a
  `mm_per_px` factor. There is no schema version and no check that IDs survive every stage.
- **Failure mode:** importance is recomputed in several places, so simplification, QA and the
  UI can disagree about what is essential.
- **Minimal change:** add optional `semantic_role` / `semantic_importance` / `render` fields
  (API stays backward-compatible), compute importance once, and add an ID-stability check
  from Model A to simplification to SVG `data-element-id`.
- **Tests:** schema round-trip (`semantic_from_dict`); IDs preserved through
  simplify/edit/regenerate; old API responses still parse.

### 2. Primitive reconstruction and editing
- **Current:** collinear fragment merging (`_merge_collinear_segments`, 15 px tolerance), vertex
  and point merging (`diagram_analysis.py`), segment grouping and closed-shape reconstruction from
  iteration 1, and the structural metric that credits a shape found as its parts.
- **Missing:** a triangle or rectangle is one polygon element; its edges are not addressable.
  The editor can retype, relabel, re-associate, nudge or delete whole elements, but cannot
  select one edge or arc. There are no fragmentation, duplicate or displacement metrics.
- **Failure mode:** one wrong vertex means deleting and losing the whole shape; duplicates and
  fragments inflate density and push sheets over the feature budget (seen on `math_061`).
- **Minimal change:** derive edge sub-IDs (`el_0.e0`...) for polygons as a view on the same
  element (not new competing objects); add a `move_vertex` edit that updates dependent
  relationships and marks them for recompute; add the three metrics first.
- **Tests:** fragments of one edge merge; two distinct adjacent edges do not; vertex edit keeps
  other elements unmoved; relationships recomputed.

### 3. Page-aware bounding-box scaling
- **Current:** `tactile_svg.page_layout` fits the **whole source image** into A4 (portrait or
  landscape, aspect preserved, 12.7 mm margins). Braille is clamped to the page;
  QA blocks `element_outside_printable_area`, `feature_below_minimum_size` and
  `exceeds_tactile_density`.
- **Missing:** the fit uses the image size, not the bounds of essential geometry plus Braille.
  Page size and margin are module constants. Braille extent is not part of the fit.
- **Failure mode:** a small figure in a large photo or worksheet is printed small, with empty paper
  around it, then fails the minimum-size check. Labels near the edge get clamped onto lines.
- **Minimal change:** compute the content box (essential geometry ∪ planned Braille boxes), scale
  it uniformly into the printable area, record source→page transform; never shrink below the
  minimum-size limits; warn and request review instead.
- **Tests:** the eight cases in the brief (small triangle, large circle, wide, tall, disconnected,
  near edge, many labels, cannot fit).

### 4. Tactile-style presets
- **Current:** `services/tactile_rules.py` `TactileRules` already holds stroke widths, spacing,
  Braille clearance, minimum feature size, density target/limit, loaded once as `TACTILE_RULES`.
  It is effectively one implicit preset.
- **Missing:** page size and margins live in `tactile_svg.py`; Braille dot geometry is constants;
  there is no named preset, and no way to recompile with different rules without re-analysis.
- **Minimal change:** move page/margin/Braille constants into one documented default preset;
  recompile endpoint takes the session's stored semantic geometry, with no OCR re-run. No
  extra presets until there are requirements.
- **Tests:** default preset reproduces today's SVG byte-for-byte; changing margin recompiles
  without calling OCR.

### 5. Bounded regions and textures
- **Current:** none. The SVG draws outlines only; there is no `<pattern>` support and no region
  model. Hatching in the source image is traced as many short lines.
- **Failure mode:** pie-chart sectors and shaded regions lose their distinction, or hatch strokes
  become noise that triggers the density block.
- **Minimal change:** detect closed faces from the line/arc graph (planar face walk), record
  region ID, boundary IDs, neighbours, uncertainty; detect source hatching as a fill of the
  region instead of lines; assign from a small set of SVG patterns (none, diagonal, dots,
  cross-hatch) so adjacent regions differ; warn when patterns run out. Open or gapped
  boundaries stay unfilled and flagged.
- **Tests:** the eight edge cases in the brief, plus a pie chart and adjacent rectangles.

### 6. Auditable simplification
- **Current:** `SimplificationAction(element_id, action, detail)`, `removed_count`,
  `merged_count`, explanations shown in the UI ("How the tactile diagram was created"),
  protected essential elements, and a blocking `density_reduction_requires_review` check.
- **Missing:** resulting IDs for merges, a fixed action vocabulary (REMOVE/MERGE/SIMPLIFY/MOVE/
  RESIZE/REASSOCIATE), reason codes, importance, stage, confidence, a review flag, and undo.
- **Known bug:** the density check counts the share of features dropped, not the printed
  result. Deleting points in the editor unblocked a dense worksheet while the SVG stayed
  byte-identical.
- **Minimal change:** extend `SimplificationAction` (backward compatible); judge density on
  the final rendered features; essential elements never removed without review.
- **Tests:** essential points, dimensions and relationships survive; every removed or merged ID
  appears in the audit; the density check reacts to output, not deletions.

### 7. Label and Braille placement
- **Current:** Liblouis translation separate from layout; `braille_layout.py` searches 9
  directions × 4 blocks for a clear position, keeps reading order, clamps to page; QA blocks
  `braille_on_line`.
- **Missing:** a combined score (distance to feature, overlap with other Braille, density,
  texture interference); no explicit "no acceptable placement" result.
- **Minimal change:** score the existing candidates deterministically; reject any with a critical
  collision regardless of score; flag the label for review when none is valid.
- **Tests:** vertex labels, angle labels, dimension text, close labels, dense figure, Braille
  beside a textured region.

### 8. QA gate
- **Current:** seven blocking checks; export is refused in both the button and `exportSvg`;
  a 0–100 readiness score; `QAReport.passes`.
- **Missing:** an explicit PASS / WARNING / BLOCKED status; separate dimensions (SVG structure,
  geometric correctness, tactile usability); checks for region consistency, texture separation
  and audit completeness.
- **Minimal change:** add `status` and per-dimension results derived from existing checks; add
  new checks only with the features above.
- **Tests:** a critical issue always gives BLOCKED regardless of score; previously passing
  fixtures still pass.

## Proposed order (one PR each, measured on dev15 + validation before and after)

0. Baseline on `main` plus the missing metrics (no pipeline change).
1. Importance/role fields, ID-stability, edge sub-IDs, `move_vertex` edit.
2. Content-box page scaling plus a default preset holding page/margin/Braille constants.
3. Regions and textures (pie charts first).
4. Auditable simplification plus the output-based density fix.
5. Braille placement scoring.
6. QA status, dimensions, regression run; held-out test split once.

Not part of this work: U-Net, deployment changes, a frontend redesign.
