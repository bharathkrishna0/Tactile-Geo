# Hosting TactileGeo with Vercel

TactileGeo has three pieces, each hosted separately:

| Piece | Host | Notes |
| --- | --- | --- |
| Frontend (React/Vite) | **Vercel** | Static build, configured by `frontend/vercel.json` |
| Backend (FastAPI) | **Render** (or any Docker host) | Configured by `render.yaml`. Needs ~4 GB RAM |
| Database + image storage | **Supabase** | Postgres and a private Storage bucket |

```
Browser ──> Vercel (static frontend)
   │
   └──HTTPS──> Render (FastAPI container) ──> Supabase Postgres + Storage
                                          └─> OpenRouter (optional, Model B)
```

## Why the backend cannot run on Vercel

Vercel Functions are limited to 2 GB of memory on Hobby and 4 GB on Pro. Python
bundles are limited to 500 MB, and request bodies to 4.5 MB. The backend needs
about 4 GB at peak on dense images. Its dependencies (PyTorch for EasyOCR, and
OpenCV) are far larger than 500 MB, and it needs the native Liblouis Braille
library from apt. Uploads can be up to 10 MB. So the backend runs as a
container, and Supabase stores its data. Supabase cannot run the Python
backend itself.

**Zero cost?** Running the backend on your laptop through a free Cloudflare
Tunnel is covered in `docs/LAPTOP_TUNNEL.md` (demo use only).

Do the steps in this order. The frontend needs the backend URL, and the backend
needs the frontend URL.

## 1. Supabase (database + storage)

1. Create a project at https://supabase.com/dashboard.
2. **Project Settings → Database → Connection string → Session pooler.** Copy
   the URI (port 5432) and fill in the password. This is `DATABASE_URL`. Use the
   pooler, not the direct connection: the direct host is IPv6-only and many
   container hosts cannot reach it.
3. **Project Settings → API.** Copy the Project URL (`SUPABASE_URL`) and the
   `service_role` key (`SUPABASE_SERVICE_ROLE_KEY`). The service-role key is
   backend-only. Never put it in Vercel or in any `VITE_*` variable.

You don't need to create any tables or buckets yourself. The backend applies
`backend/migrations/*.sql` on boot, and creates the private
`tactilegeo-sources` bucket on its first upload.

## 2. Backend on Render

1. Go to https://dashboard.render.com, then **New → Blueprint**, and pick this
   repository. Render reads `render.yaml` and proposes the `tactilegeo-api`
   service.
2. Fill in the variables it prompts for:

   | Variable | Value |
   | --- | --- |
   | `DATABASE_URL` | Supabase session-pooler URI from step 1 |
   | `SUPABASE_URL` | `https://<ref>.supabase.co` |
   | `SUPABASE_SERVICE_ROLE_KEY` | service_role key |
   | `CORS_ORIGINS` | Leave as `http://localhost:5173` for now; set it in step 4 |

3. Deploy. The first build takes several minutes because it bakes the OCR
   models into the image. When it is live, check
   `https://<service>.onrender.com/api/health`, which returns
   `{"status":"ok"}`.

The plan is `2c-4g` (2 CPU, 4 GB, a paid plan). The free and 512 MB plans run
out of memory as soon as OCR loads. `1c-2g` handles simple sheets but can run
out of memory on dense ones.

**Model B (optional):** set `MODEL_B_ENABLED=true` and add `OPENROUTER_API_KEY`
under the service's **Environment** tab. `OPENROUTER_MODEL` and
`MODEL_B_TIMEOUT_S=150` are already set for the free model that worked best in
testing; see `AGENTS.md`. Model A works without any of this.

**Other Docker hosts** (Railway, Fly.io, Cloud Run): build `backend/Dockerfile`
with `backend/` as the build context. Give it at least 4 GB of memory, the same
environment variables, and `/api/health` as the health check. The container
listens on `$PORT`.

## 3. Frontend on Vercel

1. Go to https://vercel.com/new, then **Import** this repository.
2. Set **Root Directory** to `frontend`. Vercel then picks up
   `frontend/vercel.json`: Vite, `npm ci`, `npm run build`, output `dist`.
3. Under **Environment Variables**, add:

   | Variable | Value | Environments |
   | --- | --- | --- |
   | `VITE_API_BASE_URL` | `https://<service>.onrender.com` (no trailing slash, no `/api`) | Production and Preview |

   This is inlined into the public bundle at build time, so it must be the URL
   only. If you change it later, redeploy so the build picks it up.
4. Deploy. Note the production URL, e.g. `https://tactilegeo.vercel.app`.

## 4. Allow the frontend to call the backend (CORS)

On Render, set `CORS_ORIGINS` to the Vercel production URL exactly, with no
trailing slash. Comma-separate several origins, e.g.
`https://tactilegeo.vercel.app,https://tactilegeo.example.org`. Saving restarts
the service.

Each Vercel preview deployment gets its own random URL, which is not in
`CORS_ORIGINS`. Add a preview URL there to test it, or test on production.

## 5. Check it works

1. Open the Vercel URL and upload a worksheet image.
2. Click **Process**. Preview, QA and SVG download should work. The first
   request after a deploy or restart takes 10–40 s while OCR loads.
3. In Supabase, **Table Editor** should show rows in `sessions`, and
   **Storage → tactilegeo-sources** should hold the image.
4. Restart the Render service and reopen the session. It should still be there.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Browser console says "blocked by CORS policy" | `CORS_ORIGINS` on Render doesn't exactly match the Vercel URL |
| Requests go to `https://<vercel-app>/api/...` and 404 | `VITE_API_BASE_URL` was missing at build time; set it and redeploy |
| Render shows "Out of memory" or restarts during processing | Plan smaller than `2c-4g` |
| Render log shows "could not translate host name" or "Network is unreachable" for the database | Direct Supabase connection used instead of the session pooler |
| Uploads work but disappear after a restart | `DATABASE_URL` not set, so the backend is using its in-memory store |
| AI review says Model B is unavailable | `MODEL_B_ENABLED` or `OPENROUTER_API_KEY` is not set on Render. It is never set on Vercel |

Not yet verified: this guide has not been run against a real Supabase project,
Render service, or Vercel project. The Docker image, the Postgres persistence,
and the frontend build with `VITE_API_BASE_URL` were tested locally.
