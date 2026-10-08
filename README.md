# TactileGeo

Converts educational STEM diagrams into simplified, Braille-labelled, print-ready
tactile SVG for swell-paper or embosser output, with the teacher reviewing every
step before export.

AI interprets. Computer vision measures. Fusion reconciles. The teacher decides.
The tactile compiler constructs. QA verifies. SVG delivers.

- **Model A** (deterministic OpenCV + OCR + Liblouis) is the geometry authority
  and produces the final SVG. It needs no credentials and no network.
- **Model B** (optional vision-language model via OpenRouter) interprets; fusion
  reports where it agrees, disagrees, or adds candidates for teacher review.
  Findings the teacher accepts and applies build *Semantic Geometry v2*, which
  can change types, label attachment and confirmation on Model A elements, never
  coordinates, before the tactile output is regenerated.

See `docs/VERCEL.md` for hosting on Vercel + Render + Supabase, `docs/LAPTOP_TUNNEL.md` for zero-cost demo hosting (Vercel + laptop + Cloudflare Tunnel), `docs/DEPLOYMENT.md` for production deployment, `docs/FINAL_REPORT.md` for
the architecture and measured results, and `AGENTS.md` for Model B configuration.

## Backend

Linux/macOS (Python 3.11+, Debian/Ubuntu shown):

```bash
sudo apt-get install liblouis-data python3-louis tesseract-ocr
cd backend
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
ln -s /usr/lib/python3/dist-packages/louis "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/louis"
uvicorn app.main:app --reload --port 8000
```

Windows:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Without `DATABASE_URL` sessions live in process memory (development only).
Set it to a Postgres URL and run `python scripts/migrate.py` for durable
sessions, edits, audit events, and Model B jobs.

## Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to :8000
```

## Tests and benchmark

```bash
cd backend && pytest
cd frontend && npm test && npm run typecheck && npm run build

# STEM benchmark (48 synthetic cases with exact ground truth)
cd backend
python benchmarks/stem/generate.py
python scripts/benchmark.py --out benchmarks/stem/results/after.json
```

## Braille runtime

Braille uses the official Liblouis `louis` Python bindings with the
`en-ueb-g2.ctb` table. The Docker image installs Liblouis and its tables. Windows
development needs a native Liblouis installation plus its official Python
bindings; the unrelated PyPI package named `louis` is not a Liblouis binding.

## OCR engines

EasyOCR reads all text. Tesseract is an optional second engine: it re-reads the
short label crops (vertex letters, tick numbers) that EasyOCR's detector misses,
and the two readings are fused (agreement, per-engine confidence, and whether the
text is a plausible maths label) into one label that records which engine(s)
produced it. Install the binary with `sudo apt-get install tesseract-ocr`
(Docker image: included) or, on Windows, the UB Mannheim installer
(https://github.com/UB-Mannheim/tesseract/wiki) with `tesseract.exe` on `PATH`.
If the binary is missing, the pipeline silently runs EasyOCR only.
