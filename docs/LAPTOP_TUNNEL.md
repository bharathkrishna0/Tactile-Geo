# Zero-cost hosting: Vercel + your laptop + Cloudflare Tunnel

This setup has no hosting fees and needs no card. The website is on Vercel,
and the backend runs on your own laptop. A free Cloudflare quick tunnel gives
the backend a public `https://….trycloudflare.com` address.

```
Browser ──> Vercel (static frontend, free Hobby plan)
   │
   └──HTTPS──> https://<random>.trycloudflare.com ──tunnel──> your laptop :8000 (backend)
```

It is meant for demos and judging sessions:

- It works only while the laptop is on, awake, online, and running both the
  backend and the tunnel.
- Each time you start `cloudflared`, the tunnel gets a **new random URL**, so
  you must update Vercel and redeploy (step 4). Restarting only the backend
  keeps the same URL.
- Cloudflare offers quick tunnels for testing, with no uptime guarantee and a
  limit of 200 requests at once. That is plenty for a classroom demo.

For always-on hosting, see `docs/VERCEL.md` (paid backend host).

## What the laptop needs

- About 4 GB of free RAM while processing dense images, so a laptop with 8 GB
  or more. On Windows with Docker Desktop, WSL 2 gets half the RAM by default,
  which is enough on an 8 GB+ laptop.
- Around 3 GB of free disk for the Docker image (2.4 GB with CPU-only PyTorch).
- [Docker Desktop](https://www.docker.com/products/docker-desktop/). You can
  also run the backend natively (see the README), but Docker avoids installing
  Python 3.11 and Liblouis yourself.
- `cloudflared`:
  - Windows: download the 64-bit MSI from
    https://github.com/cloudflare/cloudflared/releases/latest
    (`cloudflared-windows-amd64.msi`).
  - macOS: `brew install cloudflared`.
  - Linux: the `.deb` or `.rpm` from the same releases page.

## 1. Deploy the frontend to Vercel once

Follow step 3 of `docs/VERCEL.md`: import the repo, set **Root Directory** to
`frontend`, and deploy. For now, set `VITE_API_BASE_URL` to any placeholder,
e.g. `https://example.org`; step 4 replaces it. Note your Vercel URL, e.g.
`https://tactilegeo.vercel.app`.

## 2. Start the backend on the laptop

From the repository root, in a terminal (PowerShell, macOS Terminal, or bash):

```sh
docker build -t tactilegeo-api backend
```

The first build downloads PyTorch and the OCR models and takes 5–15 minutes.
Later builds are quicker. Rebuild after `git pull`.

```sh
docker run --rm --name tactilegeo-api -p 8000:8000 -e CORS_ORIGINS=https://tactilegeo.vercel.app tactilegeo-api
```

Replace the `CORS_ORIGINS` value with your exact Vercel URL, with no trailing
slash. Leave this terminal open. Check http://localhost:8000/api/health in a
browser; it should show `{"status":"ok"}`.

Optional extras on the same `docker run` line:

- **Keep sessions across restarts:** add
  `-e DATABASE_URL=<Supabase session pooler URI>`, which is free on Supabase's
  free plan; see step 1 of `docs/VERCEL.md`. Without it, sessions live in
  memory and are lost when the container stops. That is fine for a demo.
- **Model B:** add
  `-e MODEL_B_ENABLED=true -e OPENROUTER_API_KEY=<key> -e OPENROUTER_MODEL=dots-studio/dots-3-note-preview:free -e MODEL_B_TIMEOUT_S=150`.
  The key stays on your laptop and never reaches Vercel.

## 3. Open the tunnel

In a **second** terminal:

```sh
cloudflared tunnel --url http://localhost:8000
```

After a few seconds it prints a box containing a URL like
`https://word-word-word-word.trycloudflare.com`. Leave this terminal open too.
Check `https://<that-url>/api/health` from your phone or another network; it
should show `{"status":"ok"}`.

If it says quick tunnels are not supported, rename
`~/.cloudflared/config.yaml` (Windows: `%USERPROFILE%\.cloudflared\config.yaml`).
Cloudflare doesn't support quick tunnels while that file exists.

## 4. Point Vercel at the tunnel

1. In Vercel, open **Project → Settings → Environment Variables** and set
   `VITE_API_BASE_URL` to the tunnel URL, e.g.
   `https://word-word-word-word.trycloudflare.com`. Use no trailing slash and
   no `/api`.
2. Go to **Deployments**, then **⋯ → Redeploy** on the latest deployment. The
   variable is inlined into the page at build time, so it only takes effect
   after a redeploy.
3. Open your Vercel URL, upload a worksheet, and click **Process**.

Repeat step 4 whenever you restart `cloudflared`, because the URL changes. On
demo day, start the backend and tunnel first, then update Vercel, then share
the Vercel link.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Browser console says "blocked by CORS policy" | `CORS_ORIGINS` in `docker run` must exactly match the Vercel URL you opened. Stop the container (Ctrl+C) and rerun it with the right value |
| Requests fail with "Failed to fetch" or `ERR_NAME_NOT_RESOLVED` | The tunnel was restarted and has a new URL; redo step 4 |
| Requests go to `https://<vercel-app>/api/...` and 404 | `VITE_API_BASE_URL` was not set at build time; set it and redeploy |
| First Process takes 10–40 s | OCR models load on the first request after the backend starts; later requests take a few seconds |
| Container exits during processing | Not enough RAM. Close other apps, or raise the WSL 2 memory in `%USERPROFILE%\.wslconfig` (`[wsl2]` then `memory=6GB`) |
| Laptop went to sleep and the site stopped working | Turn off sleep while presenting. The tunnel and backend only run while the laptop is awake |
