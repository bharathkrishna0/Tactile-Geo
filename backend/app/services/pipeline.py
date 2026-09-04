from dataclasses import dataclass

from app.models.geometry import SemanticGeometry
from app.services.image_preprocessing import decode_image, preprocess_image
from app.services.vectorization import extract_shapes, shapes_to_svg
from app.services.ocr import EasyOcrProvider, OcrProvider
from app.services.label_mapping import map_label_to_geometry
from app.services.braille import LouisBrailleTranslator
from app.services.braille_layout import place_braille_markers
from app.services.image_quality import QualityReport, assess_image_quality
from app.services.diagram_analysis import analyze_diagram
from app.services.tactile_simplification import SimplifiedGeometry, simplify_geometry
from app.services.tactile_qa import QAReport, run_tactile_qa
from app.services.tactile_svg import render_tactile_svg


@dataclass
class PipelineResult:
    preview_svg: str
    tactile_svg: str | None
    shapes: list[dict]
    labels: list[dict]
    semantic_geometry: SemanticGeometry
    simplified_geometry: SimplifiedGeometry
    qa_report: QAReport
    quality_report: QualityReport


def build_full_analysis(image_bytes: bytes, edge_sensitivity: int = 50, ocr_provider: OcrProvider | None = None, braille_translator: LouisBrailleTranslator | None = None) -> PipelineResult:
    image = decode_image(image_bytes)
    quality_report = assess_image_quality(image)
    filtered = preprocess_image(image, edge_sensitivity)
    shapes = extract_shapes(filtered, edge_sensitivity)
    height, width = image.shape[:2]
    provider = ocr_provider or EasyOcrProvider()
    translator = braille_translator or LouisBrailleTranslator()
    mapped_labels = []
    for detection in provider.detect(image):
        label = map_label_to_geometry(detection, shapes)
        mapped_labels.append({**label, "braille": translator.translate(detection.text)})
    placed_labels = place_braille_markers(mapped_labels, shapes)
    svg = shapes_to_svg(shapes, width, height)
    semantic = analyze_diagram(shapes, width, height, placed_labels)
    simplified = simplify_geometry(semantic)
    qa_report = run_tactile_qa(simplified.elements, width, height)
    tactile_svg = render_tactile_svg(simplified.elements, width, height)
    return PipelineResult(
        preview_svg=svg,
        tactile_svg=tactile_svg,
        shapes=shapes,
        labels=placed_labels,
        semantic_geometry=semantic,
        simplified_geometry=simplified,
        qa_report=qa_report,
        quality_report=quality_report,
    )


def build_preview(image_bytes: bytes, edge_sensitivity: int = 50, ocr_provider: OcrProvider | None = None, braille_translator: LouisBrailleTranslator | None = None) -> tuple[str, list[dict], list[dict]]:
    image = decode_image(image_bytes)
    filtered = preprocess_image(image, edge_sensitivity)
    shapes = extract_shapes(filtered, edge_sensitivity)
    height, width = image.shape[:2]
    provider = ocr_provider or EasyOcrProvider()
    translator = braille_translator or LouisBrailleTranslator()
    mapped_labels = []
    for detection in provider.detect(image):
        label = map_label_to_geometry(detection, shapes)
        mapped_labels.append({**label, "braille": translator.translate(detection.text)})
    return shapes_to_svg(shapes, width, height), shapes, place_braille_markers(mapped_labels, shapes)
