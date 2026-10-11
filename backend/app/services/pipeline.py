from dataclasses import dataclass

from app.models.geometry import SemanticGeometry
from app.services.image_preprocessing import decode_image, preprocess_image
from app.services.vectorization import (
    drop_shapes_inside_text_regions,
    extract_shapes,
    mask_text_glyphs,
    shapes_to_svg,
    stroke_half_width,
)
from app.services.diagram_region import find_diagram_regions, inside_regions, mask_to_regions
from app.services.markers import detect_angle_arcs, detect_parallel_chevrons
from app.services.right_angles import detect_right_angle_markers, drop_marker_strokes
from app.services.ocr import EasyOcrProvider, OcrProvider
from app.services.ocr_postprocessing import postprocess_detections, restore_radicals
from app.services.label_mapping import map_label_to_geometry
from app.services.braille import LouisBrailleTranslator
from app.services.braille_layout import place_braille_markers
from app.services.image_quality import QualityReport, assess_image_quality
from app.services.image_enhancement import enhance_copy, ink_contrast_copy
from app.services.diagram_analysis import analyze_diagram
from app.services.tactile_simplification import SimplifiedGeometry, density_removed_count, simplify_geometry
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


def build_full_analysis(image_bytes: bytes, edge_sensitivity: int = 50, ocr_provider: OcrProvider | None = None, braille_translator: LouisBrailleTranslator | None = None, mask_text: bool = False) -> PipelineResult:
    image = decode_image(image_bytes)
    quality_report = assess_image_quality(image)
    # Enhance a working copy for CV/OCR; the original image is never modified.
    processing_image = enhance_copy(image)
    filtered = preprocess_image(processing_image, edge_sensitivity)
    regions = find_diagram_regions(filtered, processing_image)
    if regions:
        filtered = mask_to_regions(filtered, regions)
    height, width = processing_image.shape[:2]
    provider = ocr_provider or EasyOcrProvider()
    translator = braille_translator or LouisBrailleTranslator()
    # Text first: read labels before tracing; mask_text also erases their glyphs from the stroke mask.
    ocr_image = ink_contrast_copy(processing_image)
    raw_detections = provider.detect(ocr_image)
    detections = restore_radicals(postprocess_detections(raw_detections), ocr_image[:, :, 0])
    if regions:
        detections = [detection for detection in detections if inside_regions(detection.bbox, regions)]
    stroke_mask = mask_text_glyphs(filtered, [detection.bbox for detection in detections]) if mask_text else filtered
    shapes = extract_shapes(stroke_mask, edge_sensitivity)
    shapes = drop_shapes_inside_text_regions(shapes, [detection.bbox for detection in detections])
    stroke_half = stroke_half_width(filtered)
    text_boxes = [detection.bbox for detection in detections]
    right_angles = detect_right_angle_markers(filtered, shapes, stroke_half, text_boxes)
    angle_arcs = detect_angle_arcs(
        filtered, shapes, stroke_half, text_boxes, exclude_vertices=[marker["vertex"] for marker in right_angles],
    )
    parallel_marks = detect_parallel_chevrons(filtered, shapes, stroke_half, text_boxes)
    shapes = drop_marker_strokes(shapes, right_angles)
    mapped_labels = []
    for detection in detections:
        label = map_label_to_geometry(detection, shapes)
        mapped_labels.append({**label, "braille": translator.translate(detection.text)})
    placed_labels = place_braille_markers(mapped_labels, shapes, bounds=(width, height))
    svg = shapes_to_svg(shapes, width, height)
    semantic = analyze_diagram(
        shapes, width, height, placed_labels, right_angles=right_angles,
        angle_arcs=angle_arcs, parallel_marks=parallel_marks,
    )
    simplified = simplify_geometry(semantic)
    qa_report = run_tactile_qa(simplified.elements, width, height, density_removed=density_removed_count(simplified))
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


def build_preview(image_bytes: bytes, edge_sensitivity: int = 50, ocr_provider: OcrProvider | None = None, braille_translator: LouisBrailleTranslator | None = None, mask_text: bool = False) -> tuple[str, list[dict], list[dict]]:
    image = decode_image(image_bytes)
    processing_image = enhance_copy(image)
    filtered = preprocess_image(processing_image, edge_sensitivity)
    regions = find_diagram_regions(filtered, processing_image)
    if regions:
        filtered = mask_to_regions(filtered, regions)
    height, width = processing_image.shape[:2]
    provider = ocr_provider or EasyOcrProvider()
    translator = braille_translator or LouisBrailleTranslator()
    ocr_image = ink_contrast_copy(processing_image)
    raw_detections = provider.detect(ocr_image)
    detections = restore_radicals(postprocess_detections(raw_detections), ocr_image[:, :, 0])
    if regions:
        detections = [detection for detection in detections if inside_regions(detection.bbox, regions)]
    stroke_mask = mask_text_glyphs(filtered, [detection.bbox for detection in detections]) if mask_text else filtered
    shapes = extract_shapes(stroke_mask, edge_sensitivity)
    shapes = drop_shapes_inside_text_regions(shapes, [detection.bbox for detection in detections])
    mapped_labels = []
    for detection in detections:
        label = map_label_to_geometry(detection, shapes)
        mapped_labels.append({**label, "braille": translator.translate(detection.text)})
    return shapes_to_svg(shapes, width, height), shapes, place_braille_markers(mapped_labels, shapes, bounds=(width, height))
