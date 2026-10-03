"""Ranking and root-cause rules of the failure analysis."""

from evaluation.failure_analysis import badness, causes, severity


def score(ess_found=2, ess=4, detected=1, labels=2, fp=3, predicted=6, passes=True, blocking=(), on_text=0,
          missed=(), cer=0.0):
    obj = {"essential_found_structural": ess_found, "gt_essential": ess, "predicted": predicted,
           "structural": {"fp": fp, "recall": ess_found / ess}, "fp_reasons": {"on_text": on_text, "other": 0},
           "closed_as_sides": 0, "found_as_parts": 0, "missed_essential_ids": list(missed),
           "type_total": 0, "type_correct": 0}
    return {"objects_final": obj, "labels": {"detected": detected, "gt_essential": labels,
                                             "detection_recall": detected / labels, "cer": cer,
                                             "association_total": 0, "association_correct": 0},
            "qa": {"passes": passes, "blocking": list(blocking)}}


GT = {"objects": [{"id": "t1", "type": "tick", "semantic_importance": "essential"}]}


def test_badness_orders_failures_first():
    assert badness({"failed": True}) > badness(score(ess_found=0, passes=False))
    assert badness(score(ess_found=0)) > badness(score(ess_found=4, detected=2, fp=0))
    assert badness(score(ess_found=4, detected=2, fp=0)) == 0


def test_root_causes_from_captured_outputs():
    row = {"score": score(missed=["t1"], on_text=3, cer=0.5, passes=False, blocking=["exceeds_tactile_density"]),
           "captured": {}, "conditions": ["clean_digital"]}
    assert causes(row, GT) == ["small_features_dropped", "text_vectorised_as_geometry",
                               "ocr_misread_or_missed", "density_block"]
    timeout = {"score": {"failed": True, "error": "ImageTimeout: Model A exceeded 180 s"}, "captured": {}}
    assert causes(timeout, GT) == ["timeout_noise_explosion"]


def test_severity():
    assert severity({"score": {"failed": True}}) == "critical"
    assert severity({"score": score(passes=False)}) == "critical"
    assert severity({"score": score(ess_found=1)}) == "high"
    assert severity({"score": score(ess_found=3)}) == "medium"
