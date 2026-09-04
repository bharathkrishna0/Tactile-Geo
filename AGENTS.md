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

- `backend/` - Python FastAPI + OpenCV + EasyOCR pipeline
- `frontend/` - React + Vite + TypeScript (proxies `/api` to localhost:8000)
- `PROJECT.md` - Full product requirements document (PRD)

## Key Files

- `backend/app/services/pipeline.py` - Main CV/OCR/Braille processing pipeline
- `backend/app/services/braille.py` - Liblouis UEB Grade 2 translation
- `backend/app/core/config.py` - Environment config (upload limits, CORS)
- `frontend/src/App.tsx` - Single-file frontend (upload → preview)

## Testing

```bash
cd backend && pytest
```

Test fixtures are auto-generated in `backend/tests/fixtures/` via `conftest.py`.

## Gotchas

- Liblouis requires native installation on Windows (not PyPI `louis` package)
- Backend `uploads/` directory stores session images temporarily
- Frontend dev server runs on port 5173, proxies `/api` to backend on 8000

# Tactile Project Instructions

## Critical Project Specification

`PROJECT.md` is the primary product requirements document for this project.

Before implementing or modifying any feature:

1. Read the relevant sections of `PROJECT.md`.
2. Treat the requirements in `PROJECT.md` as mandatory.
3. Inspect referenced design images and assets when relevant.
4. Do not remove or weaken existing functionality unless explicitly requested.
5. Do not make assumptions when the specification is unclear.
6. Prefer the existing project architecture and dependencies.
7. Verify changes by running the appropriate build/test commands.

## Development Workflow

Before making significant changes:

1. Understand the existing implementation.
2. Identify the relevant files.
3. Create a clear implementation plan.
4. Implement incrementally.
5. Test the implementation.
6. Fix errors.
7. Report what changed.

## UI Requirements

When PROJECT.md references screenshots or design images:

- Treat them as visual references.
- Match layout, spacing, typography, components and interactions as closely as practical.
- Do not replace the existing design system unnecessarily.
- Ensure responsive behavior and accessibility.