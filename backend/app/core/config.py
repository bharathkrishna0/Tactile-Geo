from pathlib import Path
import os

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
UPLOAD_DIRECTORY = Path(os.getenv("UPLOAD_DIRECTORY", "uploads"))
ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if origin.strip()]

# Liblouis table path. If empty/unset, the braille service searches common
# install locations (see braille.py) for the UEB tables.
LOUIS_TABLEPATH = os.getenv("LOUIS_TABLEPATH", "").strip() or None
