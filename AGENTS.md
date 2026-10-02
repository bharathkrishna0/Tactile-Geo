# TactileGeo

Visual-to-tactile geometry conversion tool for inclusive STEM classrooms.

## Quick Commands

```bash
# Backend
cd backend && py -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm.cmd install && npm.cmd run dev

# Tests (from backend/)
pytest
```

## Architecture

Two independent layers. See `PROJECT.md` section 17 for the full boundary.

- **Model A** — deterministic OpenCV + OCR + Braille pipeline. The geometry
  authority. Owns coordinates, Braille, stroke width, printable area, the final
  SVG, and the export decision. Runs offline with no credentials.
- **Model B** — optional vision-language interpretation via OpenRouter. Advisory
  only. Returns
  regions, types, relationships, and uncertainty. Never produces geometry.
  Teacher-accepted findings reach the tactile output only through
  `semantic_v2.py`, on Model A elements, and only via the explicit apply endpoint.

- `backend/` - Python FastAPI + OpenCV + EasyOCR pipeline
- `frontend/` - React + Vite + TypeScript (proxies `/api` to localhost:8000)
- `PROJECT.md` - Full product requirements document (PRD)

## Model B environment

Model B is off unless enabled, and stays unavailable until a key is present.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_B_ENABLED` | `false` | Master switch for the advisory layer |
| `OPENROUTER_API_KEY` | *(unset)* | **Server-side only.** Never reaches the browser |
| `OPENROUTER_MODEL` | `openrouter/free` | Model or router id. `openrouter/free` picks a different model each call |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint. Override for a gateway or proxy |
| `MODEL_B_TIMEOUT_S` | `30` | Per-request provider timeout |
| `MODEL_B_MAX_CONCURRENT_JOBS` | `4` | In-flight job cap |

`OPENROUTER_API_KEY` is read from the backend process environment. Do not put it in
`.env` files that reach the frontend, in `VITE_*` variables (Vite inlines those
into the shipped bundle), or in any file committed to the repository. A test in
`backend/tests/test_model_b_live_boundary.py` fails the build if a credential
literal or a Vite-exposed key read appears anywhere in `frontend/src`.

## Live provider smoke test

Every automated test uses a fake client, so the suite needs no key, no network,
and costs nothing. One real provider call is available separately and is opt-in:

```bash
# Linux/macOS/WSL
MODEL_B_ENABLED=true OPENROUTER_API_KEY=<key> python scripts/model_b_live_smoke.py

# Windows
$env:MODEL_B_ENABLED="true"; $env:OPENROUTER_API_KEY="<key>"
python scripts/model_b_live_smoke.py
```

The script refuses to run without a key or without `MODEL_B_ENABLED=true`,
never prints the key, generates its worksheet in memory rather than writing it
to disk, and prints only aggregate counts and short quoted statements. It runs
`image -> prepare -> OpenRouter -> structured JSON -> validate -> normalise` and
deliberately does not touch Model A.

Free-tier reliability (observed 2026-10): `openrouter/free` sometimes routes to
text-only or content-safety models that return no JSON. `:free` vision models
often answer 429. `dots-studio/dots-3-note-preview:free` produced valid results
in about two of three calls, taking roughly 70 s each, so set
`MODEL_B_TIMEOUT_S=150` with it. When it fails, it has spent its output budget on
reasoning and the response is cut off. List the current free vision models with
`GET https://openrouter.ai/api/v1/models`.

Under WSL, environment variables are not passed to Windows executables by
default; add `export WSLENV=OPENROUTER_API_KEY/w:MODEL_B_ENABLED/w` if the key
appears to be ignored.

Hermetic evaluation (no key, no network) — measures the pipeline and the scoring
arithmetic, and labels itself `evidence_source: fake_client` so it can never be
quoted as a provider result:

```bash
python scripts/evaluate_model_b.py            # hermetic
python scripts/evaluate_model_b.py --live     # real calls; refuses without a key
```

## Key Files

- `backend/app/services/pipeline.py` - Main CV/OCR/Braille processing pipeline (Model A)
- `backend/app/model_b/client.py` - OpenRouter client (stdlib `urllib.request`)
- `backend/app/model_b/schema.py` - Single source of truth for the contract
- `backend/app/model_b/json_schema.py` - Strict JSON Schema derived from the models
- `backend/app/model_b/prompts.py` - Deterministic advisory prompt
- `backend/app/model_b/validator.py` - Contract enforcement; rejects, never repairs
- `backend/app/model_b/fusion.py` - Read-only reconciliation with Model A
- `backend/app/model_b/semantic_v2.py` - Semantic Geometry v2: applies teacher-accepted findings to a copy of Model A (never coordinates)
- `backend/app/model_b/service.py` - prepare -> call -> parse -> validate -> normalize
- `backend/app/services/braille.py` - Liblouis UEB Grade 2 translation
- `backend/app/core/config.py` - Environment config (upload limits, CORS, Model B)
- `frontend/src/App.tsx` - Single-file frontend (upload → preview)

## Testing

```bash
cd backend && pytest
cd frontend && npm.cmd test && npm.cmd run typecheck && npm.cmd run build
```

Test fixtures are auto-generated in `backend/tests/fixtures/` via `conftest.py`.

## Gotchas

- Liblouis requires native installation on Windows (not PyPI `louis` package)
- Backend `uploads/` directory stores session images temporarily
- Frontend dev server runs on port 5173, proxies `/api` to backend on 8000
- `google-genai` is imported lazily inside the function that needs it, so Model A
  works without the SDK installed
- `ModelBResult` deliberately has no point/radius/centre/angle field; a test
  asserts this, because it is the structural guarantee that Model B cannot
  supply a measurement
