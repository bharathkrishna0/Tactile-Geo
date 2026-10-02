# Deployment

TactileGeo has two deployable parts:

| Part | Where | Why |
| --- | --- | --- |
| Frontend (React/Vite static build) | Any static host (Vercel, Netlify, Cloudflare Pages, nginx) | No server code; talks only to the backend API |
| Backend (FastAPI + OpenCV + EasyOCR + Liblouis) | A container host (Fly.io, Render, Railway, Cloud Run, ECS, a VM) | Measured peak RSS ~4 GB on dense images and needs native Liblouis; it does not fit serverless function limits |
| Postgres (optional, recommended in production) | Supabase or any Postgres 14+ | Durable sessions, edits, audit events, Model B jobs and cache |
| Object storage (optional) | Supabase Storage private bucket | Source images outside the database; local disk otherwise |

The FastAPI backend is the only component that talks to Postgres, storage, and
the Model B provider. The browser never receives a database or provider credential.

```
Browser ──HTTPS──> static frontend
   │
   └──HTTPS /api──> FastAPI container ──> Postgres (sessions, runs, edits, audit, Model B jobs/cache)
                         │            ──> Object storage (private source images, signed URLs)
                         └──(optional)──> OpenRouter (Model B, advisory only)
```

## Backend environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Listen port inside the container |
| `WEB_CONCURRENCY` | `1` | Uvicorn workers. Each loads EasyOCR; budget ~4 GB per worker. Multiple workers are safe only with `DATABASE_URL` set |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated frontend origins |
| `DATABASE_URL` | unset | Postgres URL. Unset keeps the in-process store (development only: sessions are lost on restart and not shared between workers) |
| `RUN_MIGRATIONS` | `true` | Container applies `migrations/*.sql` on boot when `DATABASE_URL` is set. Migrations are idempotent |
| `SUPABASE_URL` | unset | With the key below, source images go to Supabase Storage |
| `SUPABASE_SERVICE_ROLE_KEY` | unset | **Backend only.** Never put it in a `VITE_*` variable |
| `SUPABASE_STORAGE_BUCKET` | `tactilegeo-sources` | Private bucket; created on first use |
| `UPLOAD_DIRECTORY` | `/data/uploads` in the image | Local storage when Supabase is not configured. Mount a volume if you rely on it |
| `SESSION_TTL_SECONDS` | `7200` | Sessions and their stored images are deleted after this. `0` keeps them |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload limit |
| `MODEL_B_ENABLED`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, ... | off | Optional advisory layer; see `AGENTS.md`. Model A works without any of them |

## Build and run the backend container

```bash
cd backend
docker build -t tactilegeo-api .

# Minimal (in-process store, local disk):
docker run --rm -p 8000:8000 -e CORS_ORIGINS=http://localhost:5173 tactilegeo-api

# Production-shaped (durable):
docker run --rm -p 8000:8000 \
  -e CORS_ORIGINS=https://tactilegeo.example.org \
  -e DATABASE_URL='postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres' \
  -e SUPABASE_URL=https://<ref>.supabase.co \
  -e SUPABASE_SERVICE_ROLE_KEY=<service-role-key> \
  tactilegeo-api

curl http://localhost:8000/api/health
```

The image bakes in the EasyOCR English models, so a cold start does not
download anything; the first request still pays the model load (~10-40 s
depending on CPU). The container runs as a non-root user and exposes a
`HEALTHCHECK` on `/api/health`.

### Supabase notes

- Use the **session pooler** connection string (port 5432) for `DATABASE_URL`.
  The backend disables server-side prepared statements, so the transaction
  pooler (port 6543) also works.
- Row-level security is enabled on every table. The backend connects as the
  database owner and enforces authorisation itself; the RLS policies restrict
  any direct `authenticated` client access to rows the user owns.
- Apply migrations manually instead of on boot with
  `RUN_MIGRATIONS=false` and `DATABASE_URL=... python scripts/migrate.py`.

### Model B workers

Model B jobs run as background tasks inside the API process, but their state is
in Postgres: a job is claimed with an atomic `queued -> running` update, so two
workers cannot run the same job, and jobs left `running` by a crashed process
are marked failed-retryable on the next request. Results are cached by image
hash + schema version + prompt fingerprint + provider + model, so repeating an
analysis of the same image does not call the provider again.

## Build and deploy the frontend

```bash
cd frontend
npm ci
VITE_API_BASE_URL=https://tactilegeo-api.example.org npm run build   # output: dist/
```

Either set `VITE_API_BASE_URL` to the backend origin (and add the frontend
origin to the backend's `CORS_ORIGINS`), or leave it unset and have the static
host rewrite `/api/*` to the backend. On Vercel the second option is a
`vercel.json` in `frontend/`:

```json
{ "rewrites": [{ "source": "/api/:path*", "destination": "https://tactilegeo-api.example.org/api/:path*" }] }
```

`VITE_API_BASE_URL` is public (Vite inlines it into the bundle); it must only
ever hold a URL, never a key.

## Local development on Linux/macOS

```bash
# Backend (Python 3.11+)
sudo apt-get install liblouis-data python3-louis    # Debian/Ubuntu
cd backend
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
# Debian packages the Liblouis binding for the system interpreter only:
ln -s /usr/lib/python3/dist-packages/louis "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/louis"
uvicorn app.main:app --reload --port 8000

# Optional durable store
docker run -d --name tg-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=tactilegeo -p 54329:5432 postgres:16
export DATABASE_URL=postgresql://postgres:postgres@localhost:54329/tactilegeo
python scripts/migrate.py

# Frontend
cd ../frontend && npm install && npm run dev        # http://localhost:5173
```

Tests: `cd backend && pytest` (the Postgres tests run when
`TEST_DATABASE_URL` is set, and skip otherwise) and
`cd frontend && npm test && npm run typecheck && npm run build`.
