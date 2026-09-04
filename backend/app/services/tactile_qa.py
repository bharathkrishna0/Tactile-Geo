from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.models.geometry import DetectedElement, GeometryType
from app.services.tactile_rules import TACTILE_RULES


@dataclass
class QAIssue:
    check: str
    severity: str
    message: str
    element_id: str | None = None


@dataclass
class QAReport:
    overall_score: float
    passes: bool
    issues: list[QAIssue] = field(default_factory=list)
    element_checks: int = 0
    score_0_100: int = 0


MIN_FEATURE_SPACING = TACTILE_RULES.minimum_feature_spacing_px
STROKE_WIDTH_MIN = TACTILE_RULES.stroke_width_min_pt
STROKE_WIDTH_MAX = TACTILE_RULES.stroke_width_max_pt
SMALL_ELEMENT_THRESHOLD = TACTILE_RULES.tiny_feature_px
COMPLEXITY_THRESHOLD = TACTILE_RULES.complexity_threshold
PRINTABLE_MARGIN_FRACTION = TACTILE_RULES.printable_margin_fraction
BRAILLE_CLEARANCE = TACTILE_RULES.braille_to_line_clearance_px
BRAILLE_SPACING = TACTILE_RULES.braille_to_braille_spacing_px


def _element_extent(element: DetectedElement) -> float:
    geo = element.geometry
    if "radius" in geo and geo["radius"]:
        return float(geo["radius"]) * 2
    points = _element_points(element)
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _element_points(element: DetectedElement) -> list[tuple[float, float]]:
    geo = element.geometry
    points: list[tuple[float, float]] = []
    if "points" in geo:
        points.extend((float(p[0]), float(p[1])) for p in geo["points"])
    if "start" in geo:
        points.append((float(geo["start"][0]), float(geo["start"][1])))
    if "end" in geo:
        points.append((float(geo["end"][0]), float(geo["end"][1])))
    if "center" in geo:
        points.append((float(geo["center"][0]), float(geo["center"][1])))
    if "position" in geo:
        points.append((float(geo["position"][0]), float(geo["position"][1])))
    if "vertex" in geo:
        points.append((float(geo["vertex"][0]), float(geo["vertex"][1])))
    if "arms" in geo:
        for arm in geo["arms"]:
            points.append((float(arm[0]), float(arm[1])))
    return points


def _distance_between_elements(a: DetectedElement, b: DetectedElement) -> float:
    if a.type is GeometryType.TEXT_LABEL and b.type is GeometryType.TEXT_LABEL:
        pa = a.geometry.get("position")
        pb = b.geometry.get("position")
        if pa and pb:
            return math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    if a.type is GeometryType.TEXT_LABEL:
        pa = a.geometry.get("position")
        if pa and _element_points(b):
            distances = [math.hypot(pa[0] - p[0], pa[1] - p[1]) for p in _element_points(b)]
            return min(distances)
    if b.type is GeometryType.TEXT_LABEL:
        pb = b.geometry.get("position")
        if pb and _element_points(a):
            distances = [math.hypot(pb[0] - p[0], pb[1] - p[1]) for p in _element_points(a)]
            return min(distances)
    pa = _element_points(a)
    pb = _element_points(b)
    best = float("inf")
    for ap in pa:
        for bp in pb:
            best = min(best, math.hypot(ap[0] - bp[0], ap[1] - bp[1]))
    return best


def run_tactile_qa(elements: list[DetectedElement], image_width: int, image_height: int) -> QAReport:
    issues: list[QAIssue] = []
    checks = 0

    if len(elements) > COMPLEXITY_THRESHOLD:
        issues.append(QAIssue(
            check="complexity",
            severity="info",
            message=f"Diagram contains {len(elements)} elements, which may be complex to feel by touch.",
        ))
    checks += 1

    for element in elements:
        extent = _element_extent(element)
        if extent and 0 < extent < SMALL_ELEMENT_THRESHOLD:
            issues.append(QAIssue(
                check="small_geometry",
                severity="warning",
                message=f"Element {element.id} is very small (about {extent:.0f}px) and may be hard to feel.",
                element_id=element.id,
            ))
        checks += 1

    for element in elements:
        bbox = element.bbox
        if bbox:
            x, y, w, h = bbox
            margin_x = image_width * PRINTABLE_MARGIN_FRACTION
            margin_y = image_height * PRINTABLE_MARGIN_FRACTION
            if x < -margin_x or y < -margin_y or x + w > image_width + margin_x or y + h > image_height + margin_y:
                issues.append(QAIssue(
                    check="printable_area",
                    severity="warning",
                    message=f"Element {element.id} extends outside the expected printable area.",
                    element_id=element.id,
                ))
            checks += 1

    # Spacing, overlap, and braille collision checks.
    labels = [e for e in elements if e.type is GeometryType.TEXT_LABEL]
    for index, a in enumerate(elements):
        for b in elements[index + 1:]:
            if a.type is GeometryType.TEXT_LABEL and b.type is GeometryType.TEXT_LABEL:
                distance = _distance_between_elements(a, b)
                if 0 < distance < BRAILLE_SPACING:
                    issues.append(QAIssue(
                        check="braille_collision",
                        severity="warning",
                        message=f"Braille markers {a.id} and {b.id} are closer than {BRAILLE_SPACING}px and may collide.",
                    ))
                checks += 1
                continue

            if a.type is GeometryType.TEXT_LABEL or b.type is GeometryType.TEXT_LABEL:
                distance = _distance_between_elements(a, b)
                if a.type is GeometryType.TEXT_LABEL:
                    clearance = BRAILLE_CLEARANCE
                else:
                    clearance = BRAILLE_CLEARANCE
                if 0 < distance < clearance:
                    issues.append(QAIssue(
                        check="braille_collision",
                        severity="warning",
                        message=f"Braille marker is closer than {clearance}px to geometry, risking indistinct touch.",
                        element_id=a.id if a.type is GeometryType.TEXT_LABEL else b.id,
                    ))
                checks += 1
                continue

            distance = _distance_between_elements(a, b)
            if distance <= TACTILE_RULES.overlap_tolerance_px:
                issues.append(QAIssue(
                    check="overlap",
                    severity="warning",
                    message=f"Elements {a.id} and {b.id} overlap, which may merge into unclear tactile shapes.",
                    element_id=a.id,
                ))
            elif 0 < distance < MIN_FEATURE_SPACING:
                issues.append(QAIssue(
                    check="spacing",
                    severity="warning",
                    message=f"Elements {a.id} and {b.id} are closer than {MIN_FEATURE_SPACING}px, which may be hard to distinguish by touch.",
                    element_id=a.id,
                ))
            checks += 1

    # Isolated elements: a geometry feature with no label and no relationship is
    # hard for a blind learner to interpret.
    for element in elements:
        if element.type is GeometryType.TEXT_LABEL:
            continue
        has_label = bool(element.associated_label_id)
        if not has_label:
            issues.append(QAIssue(
                check="isolated_element",
                severity="info",
                message=f"Element {element.id} has no associated label; consider adding a label for the learner.",
                element_id=element.id,
            ))
            checks += 1

    # High/low confidence conflict: a low-confidence feature near high-confidence
    # geometry may be noise and can confuse the diagram.
    high_confidence = [e for e in elements if e.confidence >= TACTILE_RULES.high_confidence_threshold and e.type is not GeometryType.TEXT_LABEL]
    for element in elements:
        if element.type is GeometryType.TEXT_LABEL:
            continue
        if element.confidence < 0.5:
            for strong in high_confidence:
                if _distance_between_elements(element, strong) < MIN_FEATURE_SPACING * 2:
                    issues.append(QAIssue(
                        check="confidence_conflict",
                        severity="warning",
                        message=f"Low-confidence element {element.id} sits near high-confidence geometry and may be noise.",
                        element_id=element.id,
                    ))
                    break
            checks += 1

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    total = max(1, len(issues))
    base = 1.0
    base -= (error_count * 0.4) / total if error_count else 0.0
    base -= (warning_count * 0.15) / total if warning_count else 0.0
    base -= (base * 0.1 * min(1, checks / 100))  # light complexity penalty
    overall_score = max(0.0, min(1.0, base))
    passes = error_count == 0
    return QAReport(
        overall_score=overall_score,
        passes=passes,
        issues=issues,
        element_checks=checks,
        score_0_100=round(overall_score * 100),
    )