from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.sessions import router as sessions_router
from app.core.config import CORS_ORIGINS

app = FastAPI(title="TactileGeo API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST", "PATCH"], allow_headers=["Content-Type"])
app.include_router(sessions_router)

@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
