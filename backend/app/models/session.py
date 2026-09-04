from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

@dataclass
class ConversionSession:
    session_id: str
    original_filename: str
    original_image_path: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_params: dict[str, Any] = field(default_factory=dict)
    detected_shapes: list[dict[str, Any]] = field(default_factory=list)
    detected_labels: list[dict[str, Any]] = field(default_factory=list)
    preview_svg: str | None = None
    tactile_svg: str | None = None
    quality_report: dict[str, Any] | None = None
    semantic_geometry: dict[str, Any] | None = None
    simplified_geometry: dict[str, Any] | None = None
    qa_report: dict[str, Any] | None = None
