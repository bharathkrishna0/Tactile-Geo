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
MAX_LABEL_ISOLATION_DISTANCE = 120.0
MIN_GEOMETRY_ELEMENTS = 2
DUPLICATE_COINCIDENCE_TOLERANCE = 6.0

# --- Severity policy -------------------------------------------------------
# These four checks describe tactile output a blind user physically cannot read:
# a stroke embossed outside the BANA band, braille sitting on top of a line, a
# feature running off the printable area, and a feature too small to resolve by
# touch. They are blocking, so they alone decide whether print-ready export is
# allowed. Every other check stays advisory (warning/info) and is surfaced to the
# teacher without gating export.
BLOCKING_CHECKS = frozenset({
    "stroke_width_out_of_bounds",
    "braille_on_line",
    "element_outside_printable_area",
    "feature_below_minimum_size",
})
DEFAULT_SEVERITY = "warning"

# --- Issue volume control --------------------------------------------------
# Pairwise checks are O(n^2), so a busy diagram can emit thousands of identical
# issues. That buries the actionable ones and bloats the payload, so each check
# is capped and the remainder is reported as a single aggregate issue.
MAX_ISSUES_PER_CHECK = 25
MAX_TOTAL_ISSUES = 200

# --- Readiness score -------------------------------------------------------
# The previous formula divided weighted penalties by the *total* issue count,
# which made it nearly dilution-invariant: 5000 warnings and 9999 warnings plus
# one error both scored 76/100. The score is now built from a saturating volume
# term and a separate error factor, so it is strictly decreasing in both the
# number and the severity of the issues: a noisier or more broken diagram can
# never score the same or better than a cleaner one.
SEVERITY_WEIGHTS = {"error": 4.0, "warning": 1.0, "info": 0.15}
# Weighted issue volume at which the volume term halves.
SCORE_VOLUME_HALF_POINT = 120.0
# Multiplier applied per blocking error.
SCORE_ERROR_FACTOR = 0.4


def _readiness_score(issues: list[QAIssue]) -> float:
    if not issues:
        return 1.0
    weighted = sum(SEVERITY_WEIGHTS.get(issue.severity, SEVERITY_WEIGHTS["info"]) for issue in issues)
    error_count = sum(1 for issue in issues if issue.severity == "error")
    volume = 1.0 / (1.0 + weighted / SCORE_VOLUME_HALF_POINT)
    return max(0.0, min(1.0, volume * SCORE_ERROR_FACTOR ** error_count))


def severity_for_check(check: str) -> str:
    return "error" if check in BLOCKING_CHECKS else DEFAULT_SEVERITY


class _IssueCollector:
    """Collects QA issues while bounding per-check and total volume."""

    def __init__(self) -> None:
        self._by_check: dict[str, list[QAIssue]] = {}
        self._suppressed: dict[str, int] = {}
        self.total_collected = 0
        self.total_suppressed = 0

    def add(self, check: str, message: str, element_id: str | None = None) -> None:
        bucket = self._by_check.setdefault(check, [])
        if len(bucket) >= MAX_ISSUES_PER_CHECK:
            self._suppressed[check] = self._suppressed.get(check, 0) + 1
            return
        bucket.append(QAIssue(
            check=check,
            severity=severity_for_check(check),
            message=message,
            element_id=element_id,
        ))
        self.total_collected += 1

    def issues(self) -> list[QAIssue]:
        collected = [issue for bucket in self._by_check.values() for issue in bucket]
        for check, count in sorted(self._suppressed.items()):
            collected.append(QAIssue(
                check=f"{check}_summary",
                severity="info",
                message=(
                    f"{count} further '{check}' issue{'s' if count == 1 else ''} were found "
                    f"and are not listed individually."
                ),
            ))
        # Most severe first so the export-blocking violations are never buried.
        order = {"error": 0, "warning": 1, "info": 2}
        collected.sort(key=lambda issue: order.get(issue.severity, 3))
        if len(collected) > MAX_TOTAL_ISSUES:
            remainder = len(collected) - MAX_TOTAL_ISSUES
            collected = collected[:MAX_TOTAL_ISSUES] + [QAIssue(
                check="issues_summary",
                severity="info",
                message=f"{remainder} additional issues were omitted; resolve the blocking errors first.",
            )]
        return collected


def _element_extent(element: DetectedElement) -> float:
    geo = element.geometry
    if "radius" in geo and geo["radius"]:
        return float(geo["radius"]) * 2
    # An ellipse has no radius, so without this branch its extent read as 0 and it
    # escaped the minimum-size check entirely.
    semi_axes = geo.get("semi_axes")
    if semi_axes:
        return max(float(value) for value in semi_axes) * 2
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


def _min_point_distance(point: tuple[float, float], element: DetectedElement) -> float:
    points = _element_points(element)
    if not points:
        return float("inf")
    return min(math.hypot(point[0] - p[0], point[1] - p[1]) for p in points)


def _coincident_elements(a: DetectedElement, b: DetectedElement) -> bool:
    """True when two geometry elements share nearly all their points."""
    pa = _element_points(a)
    pb = _element_points(b)
    if not pa or not pb:
        return False
    # Both extents must be comparable size.
    extent_a = _element_extent(a)
    extent_b = _element_extent(b)
    if extent_a == 0 or extent_b == 0:
        return False
    if abs(extent_a - extent_b) / max(extent_a, extent_b) > 0.5:
        return False
    # Fraction of a's points that lie near b.
    near_count = 0
    for pa_pt in pa:
        if any(math.hypot(pa_pt[0] - q[0], pa_pt[1] - q[1]) <= DUPLICATE_COINCIDENCE_TOLERANCE for q in pb):
            near_count += 1
    near_fraction = near_count / len(pa)
    return near_fraction >= 0.8


def run_tactile_qa(elements: list[DetectedElement], image_width: int, image_height: int) -> QAReport:
    collector = _IssueCollector()
    checks = 0

    if len(elements) > COMPLEXITY_THRESHOLD:
        collector.add(
            "complexity",
            f"Diagram contains {len(elements)} elements, which may be complex to feel by touch.",
        )
    checks += 1

    # Blocking: the rendered stroke width must stay inside the BANA band.
    active_stroke = float(TACTILE_RULES.stroke_width_pt)
    if active_stroke < STROKE_WIDTH_MIN or active_stroke > STROKE_WIDTH_MAX:
        collector.add(
            "stroke_width_out_of_bounds",
            (
                f"Tactile stroke width of {active_stroke:.2f}pt is outside the "
                f"{STROKE_WIDTH_MIN:.1f}-{STROKE_WIDTH_MAX:.1f}pt embossing band; "
                "lines will not be reliably readable by touch."
            ),
        )
    checks += 1

    for element in elements:
        extent = _element_extent(element)
        if extent and 0 < extent < SMALL_ELEMENT_THRESHOLD:
            collector.add(
                "feature_below_minimum_size",
                f"Element {element.id} is very small (about {extent:.0f}px) and cannot be felt reliably.",
                element_id=element.id,
            )
        checks += 1

    for element in elements:
        bbox = element.bbox
        if bbox:
            x, y, w, h = bbox
            margin_x = image_width * PRINTABLE_MARGIN_FRACTION
            margin_y = image_height * PRINTABLE_MARGIN_FRACTION
            if x < -margin_x or y < -margin_y or x + w > image_width + margin_x or y + h > image_height + margin_y:
                collector.add(
                    "element_outside_printable_area",
                    f"Element {element.id} extends outside the expected printable area.",
                    element_id=element.id,
                )
        checks += 1

    # Spacing, overlap, and braille collision checks.
    for index, a in enumerate(elements):
        for b in elements[index + 1:]:
            if a.type is GeometryType.TEXT_LABEL and b.type is GeometryType.TEXT_LABEL:
                distance = _distance_between_elements(a, b)
                if 0 < distance < BRAILLE_SPACING:
                    collector.add(
                        "braille_collision",
                        f"Braille markers {a.id} and {b.id} are closer than {BRAILLE_SPACING}px and may collide.",
                    )
                checks += 1
                continue

            if a.type is GeometryType.TEXT_LABEL or b.type is GeometryType.TEXT_LABEL:
                distance = _distance_between_elements(a, b)
                if 0 < distance < BRAILLE_CLEARANCE:
                    collector.add(
                        "braille_on_line",
                        f"Braille marker is closer than {BRAILLE_CLEARANCE}px to geometry, risking indistinct touch.",
                        element_id=a.id if a.type is GeometryType.TEXT_LABEL else b.id,
                    )
                checks += 1
                continue

            distance = _distance_between_elements(a, b)
            if distance <= TACTILE_RULES.overlap_tolerance_px:
                collector.add(
                    "overlap",
                    f"Elements {a.id} and {b.id} overlap, which may merge into unclear tactile shapes.",
                    element_id=a.id,
                )
            elif 0 < distance < MIN_FEATURE_SPACING:
                collector.add(
                    "spacing",
                    f"Elements {a.id} and {b.id} are closer than {MIN_FEATURE_SPACING}px, which may be hard to distinguish by touch.",
                    element_id=a.id,
                )
            checks += 1

    # Isolated elements: a geometry feature with no label and no relationship is
    # hard for a blind learner to interpret.
    for element in elements:
        if element.type is GeometryType.TEXT_LABEL:
            continue
        has_label = bool(element.associated_label_id)
        if not has_label:
            collector.add(
                "isolated_element",
                f"Element {element.id} has no associated label; consider adding a label for the learner.",
                element_id=element.id,
            )
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
                    collector.add(
                        "confidence_conflict",
                        f"Low-confidence element {element.id} sits near high-confidence geometry and may be noise.",
                        element_id=element.id,
                    )
                    break
            checks += 1

    # Duplicate / near-coincident geometry: two features that represent the
    # same shape overlap almost entirely, risking a double-struck tactile line.
    geometry_pairs = [(a, b) for i, a in enumerate(elements) if a.type is not GeometryType.TEXT_LABEL
                      for b in elements[i + 1:] if b.type is not GeometryType.TEXT_LABEL]
    for a, b in geometry_pairs:
        if _coincident_elements(a, b):
            collector.add(
                "duplicate_geometry",
                f"Elements {a.id} and {b.id} are nearly coincident and may represent the same feature.",
                element_id=a.id,
            )
            checks += 1

    # Dangling label: a text label whose geometry has no association and is
    # far from all geometry, so it may be stray OCR noise or an unplaced label.
    geometry_non_label = [e for e in elements if e.type is not GeometryType.TEXT_LABEL]
    for element in elements:
        if element.type is not GeometryType.TEXT_LABEL:
            continue
        if element.associated_label_id:
            continue
        geo_position = element.geometry.get("position")
        if not geo_position:
            continue
        position = (float(geo_position[0]), float(geo_position[1]))
        nearest = min((_min_point_distance(position, other) for other in geometry_non_label),
                      default=float("inf"))
        if nearest > MAX_LABEL_ISOLATION_DISTANCE:
            collector.add(
                "dangling_label",
                f"Label {element.id} is not associated with geometry and sits far from all shapes; verify it is intended.",
                element_id=element.id,
            )
            checks += 1

    # Sparse diagram: so little geometry that the image may not be a diagram.
    if len(geometry_non_label) < MIN_GEOMETRY_ELEMENTS:
        collector.add(
            "sparse_diagram",
            "Very few geometry features were detected; confirm the image is a geometry worksheet.",
        )
        checks += 1

    issues = collector.issues()
    error_count = sum(1 for issue in issues if issue.severity == "error")
    overall_score = _readiness_score(issues)
    return QAReport(
        overall_score=overall_score,
        passes=error_count == 0,
        issues=issues,
        element_checks=checks,
        score_0_100=round(overall_score * 100),
    )
