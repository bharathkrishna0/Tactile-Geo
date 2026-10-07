# TactileGeo backend: FastAPI + OpenCV + EasyOCR + Liblouis.
# Measured peak RSS is ~4 GB on dense images (EasyOCR + OpenCV); run as a container, not a serverless function.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    UPLOAD_DIRECTORY=/data/uploads \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends liblouis-data python3-louis \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-torch-cpu.txt ./
RUN pip install -r requirements-torch-cpu.txt \
    && pip install -r requirements.txt \
    # Debian ships the Liblouis binding for its own python3.11; expose only that
    # package to this interpreter instead of the whole dist-packages directory.
    && ln -s /usr/lib/python3/dist-packages/louis "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/louis" \
    && python -c "import louis; louis.translateString(['en-ueb-g2.ctb'], 'ok')"

# Bake the OCR models into the image so cold starts do not download them.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='/opt/easyocr/model')" \
    && chmod -R a+rX /opt/easyocr

COPY app ./app
COPY migrations ./migrations
COPY scripts/migrate.py ./scripts/migrate.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN useradd --system --uid 10001 tactilegeo \
    && mkdir -p /data/uploads \
    && chown -R tactilegeo /data \
    && chmod +x /usr/local/bin/docker-entrypoint.sh
USER tactilegeo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)"
ENTRYPOINT ["docker-entrypoint.sh"]
