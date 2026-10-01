"""Model B evaluation metrics.

Phase 6. Pure functions only: no network, no clock, no filesystem. Everything
here is a function of its arguments, so every number in a report can be
recomputed from the recorded inputs and a wrong metric is always a bug in the
arithmetic rather than a flaky measurement.

What is deliberately absent is a single "score". Model B is advisory, so a
composite number would invite exactly the wrong behaviour: tuning the model to
maximise one figure. The metrics are reported side by side and the failure
cases (a region claimed twice, an entity matched below threshold) are counted
explicitly rather than being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .fusion import MATCH_IOU_THRESHOLD, FusionReport

#: What a metric is computed against. Kept explicit so a report can never
#: present a fake-client number as if it came from the provider.
EvidenceSource = Literal["fake_client", "live_provider"]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two `[x, y, width, height]` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float | None:
    """`None` rather than 0.0 for an empty denominator.

    "Precision over zero predictions" is not 0%, it is undefined. Rendering it
    as 0% would make a model that correctly stays quiet look like a total
    failure, which is the opposite of what happened.
    """
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class DetectionScore:
    """Precision/recall/F1 for one label set, at a stated IoU threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    iou_threshold: float

    @property
    def precision(self) -> float | None:
        return _safe_div(self.true_positives, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float | None:
        return _safe_div(self.true_positives, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, object]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "iou_threshold": self.iou_threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def score_detections(
    predicted: Sequence[tuple[str, Sequence[float]]],
    expected: Sequence[tuple[str, Sequence[float]]],
    iou_threshold: float = MATCH_IOU_THRESHOLD,
) -> DetectionScore:
    """Greedy one-to-one match of `(label, bbox)` pairs, best IoU first.

    One-to-one for the same reason `fusion._best_overlap` is: a single
    predicted region may not confirm two separate expected regions, otherwise a
    model that boxes one shape three times scores three true positives.
    """
    candidates: list[tuple[float, int, int]] = []
    for pi, (p_label, p_box) in enumerate(predicted):
        for ei, (e_label, e_box) in enumerate(expected):
            if p_label != e_label:
                continue
            overlap = _iou(p_box, e_box)
            if overlap >= iou_threshold:
                candidates.append((overlap, pi, ei))

    # Sort by IoU descending; ties broken by index so the result is stable.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predicted: set[int] = set()
    used_expected: set[int] = set()
    for _, pi, ei in candidates:
        if pi in used_predicted or ei in used_expected:
            continue
        used_predicted.add(pi)
        used_expected.add(ei)

    true_positives = len(used_expected)
    return DetectionScore(
        true_positives=true_positives,
        false_positives=len(predicted) - len(used_predicted),
        false_negatives=len(expected) - len(used_expected),
        iou_threshold=iou_threshold,
    )


def score_relations(
    predicted: Sequence[str], expected: Sequence[str]
) -> DetectionScore:
    """Set comparison on normalised relation statements.

    Statements are compared as sets rather than sequences: order is not
    meaningful, and a model that reports the same two relationships in the
    other order has not done anything wrong. Duplicates collapse, which is the
    right call because a repeated statement adds no information.
    """
    predicted_set = {statement.strip().casefold() for statement in predicted if statement.strip()}
    expected_set = {statement.strip().casefold() for statement in expected if statement.strip()}
    true_positives = len(predicted_set & expected_set)
    return DetectionScore(
        true_positives=true_positives,
        false_positives=len(predicted_set - expected_set),
        false_negatives=len(expected_set - predicted_set),
        iou_threshold=0.0,
    )


@dataclass
class FixtureOutcome:
    """Per-fixture measurements, including every failure path."""

    fixture: str
    ok: bool = True
    latency_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    entity_score: DetectionScore | None = None
    relation_score: DetectionScore | None = None
    fusion: dict[str, int] | None = None
    evidence_source: EvidenceSource = "fake_client"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "fixture": self.fixture,
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "entity_score": self.entity_score.as_dict() if self.entity_score else None,
            "relation_score": (
                self.relation_score.as_dict() if self.relation_score else None
            ),
            "fusion": self.fusion,
            "evidence_source": self.evidence_source,
            "notes": list(self.notes),
        }


def summarise(outcomes: Sequence[FixtureOutcome]) -> dict[str, object]:
    """Aggregate outcomes into a report dictionary.

    Failed fixtures are counted, never dropped: a run where 3 of 10 requests
    errored must not report the mean of the 7 that worked without saying so.
    """
    total = len(outcomes)
    succeeded = [outcome for outcome in outcomes if outcome.ok]
    failed = [outcome for outcome in outcomes if not outcome.ok]

    latencies = [o.latency_ms for o in succeeded if o.latency_ms is not None]
    entity_tp = sum(o.entity_score.true_positives for o in succeeded if o.entity_score)
    entity_fp = sum(o.entity_score.false_positives for o in succeeded if o.entity_score)
    entity_fn = sum(o.entity_score.false_negatives for o in succeeded if o.entity_score)
    relation_tp = sum(o.relation_score.true_positives for o in succeeded if o.relation_score)
    relation_fp = sum(o.relation_score.false_positives for o in succeeded if o.relation_score)
    relation_fn = sum(o.relation_score.false_negatives for o in succeeded if o.relation_score)

    def latency_stats(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"min_ms": None, "median_ms": None, "max_ms": None}
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "min_ms": ordered[0],
            "median_ms": median,
            "max_ms": ordered[-1],
        }

    return {
        "fixtures": total,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "failure_codes": _count_by([o.error_code for o in failed]),
        "latency": latency_stats(latencies),
        "entities": {
            "true_positives": entity_tp,
            "false_positives": entity_fp,
            "false_negatives": entity_fn,
            "precision": _safe_div(entity_tp, entity_tp + entity_fp),
            "recall": _safe_div(entity_tp, entity_tp + entity_fn),
        },
        "relations": {
            "true_positives": relation_tp,
            "false_positives": relation_fp,
            "false_negatives": relation_fn,
            "precision": _safe_div(relation_tp, relation_tp + relation_fp),
            "recall": _safe_div(relation_tp, relation_tp + relation_fn),
        },
    }


def _count_by(values: Sequence[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def fusion_counts(report: FusionReport) -> dict[str, int]:
    """Counts of what reconciliation decided, for the report's agreement rate."""
    return {
        "agreements": len(report.agreements),
        "candidate_additions": len(report.candidate_additions),
        "disagreements": len(report.disagreements),
        "type_mismatches": sum(1 for d in report.disagreements if d.kind == "type_mismatch"),
        "model_a_only": sum(1 for d in report.disagreements if d.kind == "model_a_only"),
        "weak_overlaps": sum(1 for a in report.agreements if a.verdict == "weak_overlap"),
    }
