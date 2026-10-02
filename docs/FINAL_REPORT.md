# TactileGeo — Final Engineering Report

Branch `optimization/dual-architecture`, measured against `main` at `eba4fd8`.

> AI interprets. Computer vision measures. Fusion reconciles. The teacher
> decides. The tactile compiler constructs. QA verifies. SVG delivers.

All numbers below come from `backend/scripts/benchmark.py` (results committed in
`backend/benchmarks/stem/results/`) or from the test suites. "Before" is the
same scorer run against the `main` checkout. Nothing here is a claim of BANA,
ICEB or any other certification; the tactile thresholds are conservative,
guideline-informed defaults.

---

## 1. Architecture diagram

```
 Image
   ├─> Model A (deterministic CV, OCR, Braille) ── geometry authority ─────────┐
   └─> Model B (optional, async, semantic AI)                                   │
             │                                                                  v
             └──────────────> FUSION ENGINE (read-only reconcile, per-entity review)
                                                │
                              teacher accept / reject / defer (audit events)
                                                │  explicit "Apply accepted findings"
                                                v
                      Semantic Geometry v2 = Model A geometry + accepted findings
                      (type corrections, Model A labels attached, confirmations;
                       coordinates always Model A's, reversible, idempotent)
                                                │
                                                v
                 Tactile simplification ─> Braille layout ─> QA engine (mm)
                                                │
                                                v
                     Teacher approval (export gate, no bypass) ─> FINAL SVG

 Persistence (Postgres + object storage): sessions, runs, edits, audit events,
 Model B jobs + content-hash cache. Teacher edits re-enter at simplification.
```

## 2. Model A architecture

`decode -> quality report -> enhance copy (CLAHE, sharpen, polarity) -> preprocess
-> region mask -> contour/segment extraction -> OCR -> label mapping -> Liblouis
UEB grade 2 -> Braille placement -> semantic analysis -> tactile simplification
-> tactile QA (mm) -> tactile SVG`. Deterministic, offline, credential-free.

| Change | Before | After | Why | Measurement | Test |
|---|---|---|---|---|---|
| Polarity normalisation | Light-on-dark pages thresholded as dark ink on light paper; double contours and spurious polygons | `normalize_polarity()` inverts when the page median is dark and ink sits in the bright tail | Chalkboards, inverted scans | Dark-variant F1 0.559 -> 0.826; dark-variant false positives removed in 9/12 figures | `test_physical_tactile_qa.py` dark-page cases |
| Density budget in simplification | No cap; 326-feature screenshot passed | Trims to 40 features, protects labelled/right-angle/teacher-confirmed elements, logs each removal | Over-dense sheets are unreadable by touch | Reference screenshot: 326 tactile features / pass -> 100 / blocked for review | `test_tactile_qa_extended.py` |

## 3. Model B architecture

Optional OpenRouter vision-language call through stdlib `urllib`. Strict JSON
Schema, validator that rejects and never repairs, `ModelBResult` with no
coordinate/radius/angle fields. Now durable:

| Change | Before | After | Why | Measurement | Test |
|---|---|---|---|---|---|
| Job store | Process-local dict; restart loses jobs; two workers can both run a job | Postgres `model_b_jobs`, atomic `queued->running` claim, atomic cancel, stale `running` jobs failed-retryable on recovery | Cloud deploys restart and scale out | Restart and race scenarios covered | `test_model_b_durable.py` (12), `test_model_b_api.py` |
| Result cache | None; every request calls the provider | Key = sha256(image) + schema version + prompt fingerprint + provider + model; Postgres table or bounded in-memory LRU (200) | Cost and latency; invalidates when any input to the answer changes | Cache hit returns without a provider call; `cache_hit` exposed in API/UI | cache hit/miss/invalidation tests |
| Lossless serialisation | Result objects only in memory | `result_codec` round-trips nested entities/enums | Needed to persist results | Round-trip equality | codec test |

## 4. Fusion architecture

`reconcile(model_a, model_b)` remains pure. New `EntityReview` per Model B
entity: `model_a_id`, `correspondence` (strong/weak/none), `adds_semantics`,
`contradicts_model_a`, `requires_review`, `reason`, `advisory_only=True`.

| Before | After | Why | Measurement | Test |
|---|---|---|---|---|
| Aggregate agreement/disagreement lists only | Per-entity answers plus persisted teacher decisions (`accept`/`reject`/`defer`) as audit events with `applied_to_geometry: false` | The teacher needs to act per finding; Model B must never change geometry | Every Model B entity gets exactly one review | `test_model_b_fusion.py`, decision endpoint tests (unknown finding -> 422) |
| Accepted findings never reached the tactile output | `build_semantic_geometry_v2()` (`app/model_b/semantic_v2.py`) applies accepted findings to a copy of Model A's semantic geometry, then simplification, Braille, QA and SVG are regenerated | Teacher-approved semantics should shape the sheet, while Model A stays the geometry authority | Coordinates and bboxes byte-identical to Model A in every test; Model B-only regions add no element | `test_semantic_geometry_v2.py` (24) |

Semantic Geometry v2 rules: only `accept` decisions apply; only findings that
correspond to a Model A element apply. Three changes exist: `set_type` (only
when Model A's own geometry can draw the new type, e.g. a triangle needs three
Model A vertices), `attach_label` (associates an existing, unattached Model A
text label whose text matches Model B's reading; no label is created) and
`confirm` (teacher override, which protects the element from density
simplification). Each change stores the previous values on the element, so a
withdrawn acceptance is restored on the next apply unless the teacher has since
edited the element by hand. The fusion endpoint compares against the
pre-application baseline, so applying does not hide the reviews it acted on.

## 5. Database architecture

`migrations/001_persistence.sql`: `profiles`, `projects`, `sessions` (JSONB
document + `source_storage_key`), `processing_runs`, `teacher_edits`,
`audit_events`. `002_model_b_jobs.sql`: `model_b_jobs`, `model_b_cache`. Files
live in object storage; Postgres stores keys. RLS on every table, policies keyed
to `auth.uid()` when present. TTL cleanup deletes rows and stored objects.
`scripts/migrate.py` is idempotent (`schema_migrations`).

| Before | After | Why | Measurement | Test |
|---|---|---|---|---|
| In-process sessions, local disk | `PostgresSessionStore` when `DATABASE_URL` is set, in-memory fallback otherwise; `ObjectStorage` (local or private Supabase bucket, created on first upload) | Restart survival, multi-worker | Session survives a new store instance | `test_postgres_store.py` (migration idempotency, round-trip, restart, runs, edits, audit, TTL, RLS) |

## 6. Cloud deployment architecture

Static frontend (any CDN) -> FastAPI container -> Postgres/Supabase + private
storage -> optional OpenRouter. The backend's measured peak RSS is ~4 GB, so it
runs as a container, not a serverless function. See `docs/DEPLOYMENT.md`.

| Before | After | Why |
|---|---|---|
| Dockerfile copied only `app/`, put all of Debian dist-packages on `PYTHONPATH`, downloaded OCR models at first request, ran as root | Liblouis binding linked alone, OCR models baked in, migrations + entrypoint, non-root user, `HEALTHCHECK`, `PORT`/`WEB_CONCURRENCY` | Reproducible cold starts, safe defaults |
| Frontend could only call a same-origin `/api` | `VITE_API_BASE_URL` (public URL only) or host rewrite | Frontend and backend on different hosts |

## 7. API changes

- `POST /api/sessions/{id}/model-b/{job_id}/decisions` — record a teacher decision on a Model B finding (201).
- `POST /api/sessions/{id}/model-b/{job_id}/apply` — build Semantic Geometry v2 from accepted findings and regenerate; returns the session plus a per-finding outcome (`applied`, `change`, `detail`) and the withdrawn ids. Recorded as a `model_b_apply` run and a `model_b_applied` audit event; 409 before Model A has run.
- `GET .../model-b/{job_id}` adds `cache_hit`; fusion report adds `entity_reviews` and `decisions`.
- Missing source image after expiry -> HTTP 410 instead of 500.
- QA report adds `exceeds_tactile_density`, `density_reduced`, `density_reduction_requires_review`; existing checks are now in mm.
- All existing endpoints and response shapes are otherwise unchanged.

## 8. Frontend changes

Model B panel: cache-reuse message, per-entity review queue showing only
findings that need review, Accept / Reject / Decide later buttons, restored
server decisions, and an "Apply accepted findings to the tactile output"
button (shown once something is accepted) that swaps in the regenerated session
and lists what was and was not applied. Export gate knows the two new blocking
density checks.
`API_BASE` for cross-origin deployment.

## 9. Accessibility improvements

Native buttons with `aria-pressed`, visible focus, `aria-live` count of
remaining reviews, `role="alert"` on save and apply failures, an `aria-live`
summary of applied findings, and a plain-language note that accepting changes
the embossed output only after applying, and only where Model A has geometry. Covered by Testing
Library keyboard tests and `jest-axe` (no violations).

## 10. Tactile improvements

| Before | After | Measurement |
|---|---|---|
| Thresholds in source pixels; same drawing passed or failed depending on upload resolution | Thresholds in mm on the A4 page via the compiler's own `page_layout().mm_per_px`: feature >= 5 mm, spacing >= 6 mm, overlap tolerance 1.5 mm | Resolution-independent by construction; QA pass rate 0.875 -> 0.958 on the benchmark |
| No density limit | > 40 features warns, > 60 blocks, heavy trimming requires teacher review | 4 of 7 reference images now blocked for review instead of exporting unreadable sheets |

## 11. Braille improvements

Clearance measured from the embossed cell block (2.5 mm dot pitch, 6 mm cell
pitch), not the label centre: >= 3 mm to lines, >= 3 mm between labels.
Labels placed with the same geometry QA uses. Braille-on-line false passes and
false failures fixed (cell figure: 3 of 4 variants blocked before, all pass after).
OCR label recall on the benchmark 0.556 -> 0.611.

## 12. QA improvements

Physical-unit checks, density checks, segment-to-rectangle intersection for
Braille-on-line, and an explicit "requires review" block when simplification
removes more than it keeps. Export stays gated on blocking checks in both the
API and the UI.

## 13. Performance before/after

Same host, both runs concurrent, EasyOCR warm (first call excluded).

| Metric | Before | After |
|---|---|---|
| Median Model A time per image | 7.91 s | 7.92 s |
| p95 | 12.24 s | 11.83 s |
| Peak RSS (whole run incl. dense references) | 3914 MB | 3922 MB |

Model A latency is unchanged; the PRD's 8 s target is met at the median and
missed at p95. Durable storage adds one Postgres round trip per request; the
Model B cache removes the provider call entirely on a repeat image.

## 14. Accuracy before/after

48 synthetic cases (12 figures × clean, half-resolution, phone-photo, dark page)
with exact ground truth.

| Metric | Before | After |
|---|---|---|
| Shape precision | 0.627 | 0.784 |
| Shape recall | 0.784 | 0.784 |
| Shape F1 | 0.697 | 0.784 |
| Type accuracy (of found shapes) | 0.884 | 0.870 |
| Label recall | 0.556 | 0.611 |
| QA pass rate | 0.875 | 0.958 |
| Pipeline error rate | 0.021 | 0.021 |

Type accuracy dips because three dark-page triangles that were previously
found as (noisy) triangles are now found as their three sides, which is how the
clean versions of the same figures are already represented.

## 15. Benchmark results

| Slice | F1 before | F1 after | QA pass before | QA pass after |
|---|---|---|---|---|
| clean | 0.869 | 0.869 | 0.833 | 1.0 |
| photo | 0.889 | 0.889 | 0.917 | 1.0 |
| dark | 0.559 | 0.826 | 0.917 | 1.0 |
| half-resolution | 0.513 | 0.513 | 0.833 | 0.833 |
| geometry | 0.657 | 0.738 | 0.958 | 0.958 |
| physics | 0.631 | 0.727 | 1.0 | 1.0 |
| graphs | 0.889 | 0.980 | 0.875 | 1.0 |
| chemistry | 0.353 | 0.400 | 1.0 | 1.0 |
| biology | 0.750 | 0.857 | 0.0 | 0.75 |

Reference images (real uploads, no ground truth): see
`benchmarks/stem/results/comparison.md`. 3 of 7 export; the 4 dense ones are
blocked with `density_reduction_requires_review` instead of exporting 74-326
raised features.

Reproduce: `python benchmarks/stem/generate.py && python scripts/benchmark.py --out after.json`;
`--backend-root <other checkout>` measures another commit; `--compare a.json b.json` prints the tables.

## 16. Tests

| Suite | Before | After |
|---|---|---|
| Backend (`pytest`, with `TEST_DATABASE_URL`) | 588 | 638 passed, 0 skipped |
| Frontend (Vitest) | 82 | 91 passed; typecheck and build pass |

New: physical QA, density budget, object storage (incl. bucket creation),
Postgres store, durable Model B (claims, cancellation races, recovery, cache),
per-entity fusion, decisions endpoint, review-queue keyboard and axe tests.
The structural test that Model A services never import Model B still passes.

## 17. Remaining limitations

- **Half-resolution uploads** are the weakest slice (F1 0.513); one case
  (`two_circles` at 500×400) detects no geometry and the API returns 422
  "Cannot map labels because no vector geometry was detected".
- **Regular polygons** (pentagon, hexagon) are not recognised as polygons;
  chemistry F1 is 0.40.
- **Dense real worksheets** are blocked rather than automatically split; the
  teacher must crop or split them.
- **Supabase cloud path** (storage bucket, pooler, RLS under `auth.uid()`) is
  implemented and unit-tested with mocked HTTP, and RLS was tested on local
  Postgres, but has not been exercised against a live Supabase project because
  credentials were not available in this session.
- **Model B live accuracy** is not measured here; all automated tests use a fake
  provider. Decisions are recorded but do not yet feed back into Model A edits.
- **Latency** p95 ~12 s exceeds the 8 s target; peak memory ~4 GB.
- Synthetic benchmark figures are clean by construction and do not replace
  evaluation on real classroom worksheets.
- No authentication layer: `profiles`/`projects` and RLS exist, but the API does
  not yet issue or check user identity.

## 18. Deployment instructions

See `docs/DEPLOYMENT.md`. Short form:

```bash
cd backend && docker build -t tactilegeo-api .
docker run -p 8000:8000 -e CORS_ORIGINS=https://<frontend> \
  -e DATABASE_URL=<postgres url> -e SUPABASE_URL=... -e SUPABASE_SERVICE_ROLE_KEY=... tactilegeo-api
cd frontend && VITE_API_BASE_URL=https://<backend> npm run build   # deploy dist/
```
