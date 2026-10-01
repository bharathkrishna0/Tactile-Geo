  # Tier 1 + Tier 2 Implementation Checkpoint

  Status: **Tier 1 + Tier 2 complete and verified.**
  Last verified: backend `147 passed` (`.venv/Scripts/python.exe -m pytest -q`);
  frontend `25 passed` (`npm test`) with clean `npm run typecheck` and `npm run build`.

  Scope is Tier 1 + Tier 2 only. **No Gemini / Model B work was started**, per instruction #1,
  even though `PROJECT.md` now contains a "Gemini Dual-Architecture Integration" spec.

  ---

  ## Completed and verified

  ### 1. Angle heuristic — option (a), tighten the gate
  `backend/app/services/segment_geometry.py` (NEW) holds the collinearity primitives, shared by
  vectorization and simplification so the two rule sets cannot drift apart.

  - `vectorization.py`: Hough output is deduplicated + collinear-merged **before** it becomes a
    shape. Triangle fixture went from **12 line shapes to 3**.
  - `diagram_analysis.py::_detect_angles` now enforces:
    1. both arms must be real edges (`MIN_ARM_LENGTH_PX = 12.0`), so coincident fragments and
      sub-pixel stubs are rejected;
    2. **at most one angle per vertex** (longest arms win), so angle count can never exceed the
      vertex count. Hard ceiling `MAX_ANGLE_ELEMENTS` as a second guard.
  - Fixed a genuine pre-existing bug: the arm endpoint was chosen by *equality* with the shared
    vertex. Any segment pair meeting start-to-start or end-to-end therefore produced a
    zero-length arm and the angle was silently lost. Now uses `_far_endpoint_from()`.

  **Measured on `triangle_worksheet.png`:** 96 semantic elements / 78 angles → **8 elements /
  3 angles**, at the 3 genuine triangle vertices (56.4°, 61.9°, 61.2°).

  ### 2. QA severity policy
  `tactile_qa.py` now defines `BLOCKING_CHECKS` = the four requested checks; `severity_for_check()`
  returns `error` for them, `warning` for everything else. Renamed to match the spec:

  | old check | new check | severity |
  |---|---|---|
  | `small_geometry` | `feature_below_minimum_size` | error (blocking) |
  | `printable_area` | `element_outside_printable_area` | error (blocking) |
  | `braille_collision` (label↔geometry) | `braille_on_line` | error (blocking) |
  | `braille_collision` (label↔label) | `braille_collision` | warning |
  | — | `stroke_width_out_of_bounds` | error (blocking) |

  `stroke_width_out_of_bounds` is a *real* check, not a stub: `tactile_rules.py` gained
  `stroke_width_pt` (the value actually used for rendering, default 2.0) and QA validates it
  against the BANA band. `tactile_svg.py` now renders `TACTILE_RULES.stroke_width_pt` instead of
  hardcoding the max, so the check validates the rendered value.

  ### 3. QA score formula — was dilution-invariant
  Old: penalties divided by total issue count, so 1..5000 warnings all scored 76/100, and
  9999 warnings + 1 error also scored 76/100.
  New `_readiness_score()` = saturating volume term × per-error factor. Verified monotone:

  | issues | score |
  |---|---|
  | none | 100 |
  | 1 warning | 99 |
  | 10 warnings | 92 |
  | 50 warnings | 71 |
  | 5000 warnings | 2 |
  | 9999 warnings + 1 error | 0 |

  ### 4. QA issue volume — was 5151 issues on the triangle
  Added `_IssueCollector`: caps each check at `MAX_ISSUES_PER_CHECK = 25` and the total at
  `MAX_TOTAL_ISSUES = 200`, emits a `*_summary` info issue for the remainder, and sorts errors
  first so blocking violations are never buried.

  ### 5. OCR — real EasyOCR now reads the demo fixtures
  `ocr.py`: images are upscaled (`ocr_scale_factor`, target 960px short side, ceiling 2400px) before
  `readtext`, and bounding boxes are divided back to original-image coordinates.
  The reader is now cached per language set in a module-level dict + lock, so the ~10s model load
  happens once per process instead of per request.

  `labelled_triangle_worksheet.png` went from **0 labels to `A`, `B`, `C`**.

  ### 6. Text glyphs no longer become geometry
  New `vectorization.drop_shapes_inside_text_regions()`, called in `pipeline.py` after OCR.
  A letter glyph has enough closed area to pass the contour filter, so labels were being
  vectorized twice — once as braille, once as a tiny polygon. That polygon then tripped the new
  blocking `feature_below_minimum_size` check and blocked export on a correctly processed
  worksheet.

  ### 7. Simplification no longer drops elements silently
  Previously, an exactly-duplicate collinear segment was marked `used` but never kept and never
  reported, so elements vanished with no `SimplificationAction`. Now duplicates merge and are
  recorded (`merged_from` semantic property, `merged_count`, an explanation naming the absorbed
  ids). Added a hard invariant: every input element must be kept, removed as noise, or explicitly
  consumed by a merge — otherwise a `dropped_unexplained` action is emitted.

  ### 8. Tactile SVG actually emits braille
  `render_tactile_svg` was writing plain OCR text (`class="braille"` but the *content* was `"A"`),
  so the "print-ready" export was not tactile. It now emits `geometry["braille"]` (falling back to
  text), adds `font-family` with a U+2800-capable stack, `aria-label` with the original text, and
  runs everything through a new `sanitize_svg()` that strips `<script>`, `<foreignObject>`,
  `on*=` handlers and `javascript:` URLs — the SVG is rendered via `dangerouslySetInnerHTML`.

  ### 9. Ellipse extent bug
  `_element_extent` had no `semi_axes` branch, so every ellipse read as extent 0 and escaped the
  minimum-size check entirely. Fixed.

  ---

  ## Known remaining failure

  `tests/test_demo.py::test_demo_result_survives_edit_regeneration`

  `RuntimeError: Can't translate: tables ['en-ueb-g2.ctb'], inbuf A`

  This is an **environment/pre-existing** problem, not a regression in the new code. It is simply
  the first time the real translator is exercised on this path — before the OCR fix, no text label
  was detected, so no translation was attempted (the test failed earlier with a `422`).

  Tables exist at `backend/.native/win64/share/liblouis/tables/`, but
  `braille.resolve_tablepath()` only probes Linux/Unix paths, so it returns `None` and the `louis`
  bindings fall back to a non-existent path.

  **Next action:** add the repo-local Windows location to `_DEFAULT_TABLEPATHS` in
  `backend/app/services/braille.py`, resolved relative to the backend package root rather than
  hardcoded. Candidates to probe:
  `Path(__file__).resolve().parents[3] / ".native" / "win64" / "share" / "liblouis" / "tables"`.
  Do not add a repo-specific absolute path.

  ---

  ## Not started

  - **Frontend export interlock** — `App.tsx::exportSvg` still has the "Export anyway" bypass
    (`if (critical.length > 0 && exportWarning === null)`), so export is *not* genuinely blocked.
    The interlock must be made real; the user allowed the minimal user-visible change needed for
    the gate to be truthful and enforceable.
  - **Contrast token** — `#787774` is still in `frontend/src/styles.css` (lines 18 and 140) and
    fails WCAG AA on white. Needs darkening to >= 4.5:1 plus a regression check.
  - **Frontend tests** — vitest, `@testing-library/react`, jest-axe not installed. No tests written.
  - **Verification still owed:** frontend typecheck, frontend build, frontend tests, and the final
    before/after report the user asked for (element counts, angle counts, QA counts, readiness
    score, export behaviour).
  - **SVG texture patterns** (raised in the review, not yet decided) — no `<pattern>` in the
    tactile output; BANA-style fill differentiation is still absent.

  ---

  ## Files changed in this phase

  New:
  - `backend/app/services/segment_geometry.py`
  - `backend/app/services/tactile_qa.py` — policy, caps, score, stroke-width check, ellipse extent
  - `backend/app/services/tactile_simplification.py` — consumed-id accounting, duplicate reporting, shared helpers
  - `backend/app/services/tactile_svg.py` — braille output, font stack, sanitization
  - `backend/app/services/ocr.py` — upscaling, reader cache
  - `backend/app/services/vectorization.py` — segment normalization, text-region suppression
  - `backend/app/services/diagram_analysis.py` — angle gate, far-endpoint fix
  - `backend/app/services/tactile_rules.py` — `stroke_width_pt`
  - `backend/app/services/pipeline.py` — wire text-region suppression
  - `backend/tests/test_tactile_simplification.py` — +2 regression tests, helper made non-collinear

  Modified tests (contract renames + faithful fake reader):
  - `backend/tests/test_tactile_qa.py`, `test_phase2_semantics.py`, `test_ocr_braille_and_labels.py`

  Do not reset or discard the other pre-existing uncommitted changes in the working tree
  (`.native/**`, `PROJECT.md`, several `frontend/` files, `conftest.py`, etc.).

  ---

  ## Final verification (added after the initial checkpoint)

  ### Braille translation now works for real
  `braille.py` resolves an absolute table path (repo-local `.native/win64/share/liblouis/tables/`,
  then the loaded native library's own tables) because the bundled liblouis 3.39 ignores
  `LOUIS_TABLEPATH`. `to_unicode_braille()` maps liblouis ASCII output to real U+2800–U+28FF
  cells. Real translation: `A` → `⠬⡡`, `Triangle` uses the UEB contraction.

  ### Export interlock is now a genuine block
  - **NEW** `frontend/src/lib/exportGate.ts` — single source of truth mirroring the backend
    `BLOCKING_CHECKS`. An issue blocks if `severity === 'error'` **or** its `check` is one of the
    four blocking names, so the gate also holds against an older backend that downgraded them.
  - `App.tsx`: the "Export anyway" bypass is **removed**. The button is `disabled` when blocked,
    carries `aria-describedby`, and `exportSvg()` independently refuses to build a Blob, so no
    code path can emit a rejected file. The blocked panel lists the offending issues and links
    to the views where they can be fixed.

  ### Contrast
  `#787774` measured **4.478:1** on white — a genuine AA failure. Introduced
  `--color-text-secondary: #6e6d6b` (**5.17:1**) and applied it to `.muted` and the drop-zone
  hint text. Guarded by `src/styles.test.ts`, which also asserts the old value never reappears.

  ### Frontend test infrastructure
  Dev-only additions (runtime deps untouched): `vitest`, `@vitest/coverage-v8`,
  `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`,
  `jest-axe`, `@types/jest-axe`, `jsdom`. Config in `vitest.config.ts`, setup in
  `src/test/setup.ts`. Added `typecheck` and `test` npm scripts.

  Also fixed two latent build-hygiene defects: `vite.config.js`, `vite.config.d.ts` and both
  `*.tsbuildinfo` files were **tracked in git** and re-emitted on every build, so `npm run build`
  always dirtied the working tree. They are now gitignored, untracked, and
  `tsconfig.node.json` emits into `node_modules/.tmp`.

  ### A11y defect fixed
  The visually-hidden file input had no accessible name; it now has
  `aria-label="Choose a worksheet image"`.

  ## Real-pipeline results (all fixtures, `scripts/pipeline_metrics.py`)

  | fixture | raw | semantic | angles | simplified | QA score | errors | warnings |
  |---|---|---|---|---|---|---|---|
  | triangle | 5 | 9 | 3 | 9 | 69 | 0 | 54 |
  | circle | 2 | 2 | 0 | 2 | 98 | 0 | 2 |
  | ellipse | 5 | 8 | 0 | 8 | 89 | 0 | 15 |
  | complex | 6 | 6 | 0 | 6 | 95 | 0 | 6 |
  | labelled_triangle | 3 | 7 | 1 | 7 | 94 | 0 | 8 |
  | blurred | 11 | 15 | 4 | 15 | 73 | 0 | 44 |
  | dark | 0 | 0 | 0 | 0 | 99 | 0 | 1 |

  **Zero blocking errors on every fixture**, so print-ready export is available for all of them.
  `labelled_triangle_worksheet.png` reads **A, B, C** and emits 8 real braille cells in the SVG.

  ## Remaining limitations
  1. **No texture patterns.** Every element is a uniform raised line/shape; `PROJECT.md` requires
    distinct textures (e.g. triangle = smooth) to disambiguate visually identical outlines. Not
    in the Tier 1/2 scope and not started.
  2. **`dark_worksheet.png` yields zero elements.** The image-quality report flags the gate, but
    the pipeline still extracts nothing, so there is nothing for a teacher to correct. Better
    low-light preprocessing is outstanding.
  3. **`blurred_worksheet.png` is noisy** (10 line segments, 4 angles from 11 raw shapes) and
    scores 73. Blur-specific shape rejection would help.
  4. **Angle enum naming.** Semantic `type` values are `GeometryType` enums, not the bare strings
    the metrics script normalises to; anything consuming the API should not assume strings.
  5. **`scripts/pipeline_metrics.py` is a reporting tool**, not a test. It is not wired into CI.
