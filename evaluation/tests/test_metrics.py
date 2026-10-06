"""Arithmetic tests for the metrics engine on hand-built cases."""

from evaluation import metrics
from evaluation.aggregate import summarize


def seg(oid, a, b, typ="line_segment", importance="essential"):
    return {"id": oid, "type": typ, "geometry": {"kind": "segment", "points": [list(a), list(b)]},
            "semantic_importance": importance}


def tri(oid, pts):
    return {"id": oid, "type": "triangle", "geometry": {"kind": "polygon", "points": [list(p) for p in pts]},
            "semantic_importance": "essential"}


def pline(pid, a, b, typ="line_segment"):
    return {"id": pid, "type": typ, "geometry": {"start": list(a), "end": list(b)}}


def ppoly(pid, pts, typ="triangle"):
    return {"id": pid, "type": typ, "geometry": {"points": [list(p) for p in pts]}}


def gt_doc(objects, labels=(), points=(), relationships=()):
    return {"width": 1000, "height": 800, "objects": list(objects), "labels": list(labels),
            "points": list(points), "relationships": list(relationships)}


TRI = [(100, 100), (500, 100), (300, 400)]


def test_prf_arithmetic():
    r = metrics.prf(3, 1, 2)
    assert r["precision"] == 0.75 and r["recall"] == 0.6
    assert r["f1"] == round(2 * 0.75 * 0.6 / 1.35, 4)
    assert metrics.prf(0, 0, 0)["f1"] is None
    assert metrics.prf(0, 2, 2)["f1"] == 0.0


def test_levenshtein_and_normalization():
    assert metrics.levenshtein("kitten", "sitting") == 3
    assert metrics.levenshtein(["a", "b"], ["a", "c", "b"]) == 1
    assert metrics.normalize_text("  5 cm\u2032 ") == "5 cm'"


def test_percentiles():
    p = metrics.percentiles([1, 2, 3, 4])
    assert p["p50"] == 2.5 and p["n"] == 4 and p["mean"] == 2.5
    assert metrics.percentiles([]) is None


def test_exact_segment_match_and_errors():
    gt = gt_doc([seg("s1", (100, 100), (500, 100))])
    s = metrics.score_objects(gt, [pline("e1", (502, 101), (98, 99))])
    assert s["strict"]["tp"] == 1 and s["strict"]["fp"] == 0
    geo = s["geometry_errors"]
    assert geo["endpoint_px"][0] < 3
    assert geo["angle_deg"][0] < 0.5
    assert s["type_correct"] == 1


def test_far_or_partial_segment_is_not_a_match():
    gt = gt_doc([seg("s1", (100, 100), (900, 100))])
    s = metrics.score_objects(gt, [pline("e1", (100, 100), (300, 100))])
    assert s["strict"]["tp"] == 0 and s["strict"]["fn"] == 1 and s["strict"]["fp"] == 1


def test_broken_line_counts_as_parts_not_strict():
    gt = gt_doc([seg("s1", (100, 100), (900, 100))])
    s = metrics.score_objects(gt, [pline("a", (100, 100), (495, 100)), pline("b", (505, 100), (900, 100))])
    assert s["strict"]["tp"] == 0
    assert s["structural"]["tp"] == 1 and s["structural"]["fp"] == 0
    assert s["fragmented_linear"] == 1


def test_triangle_from_three_sides_counts_as_parts():
    gt = gt_doc([tri("t1", TRI)])
    sides = [pline("a", TRI[0], TRI[1]), pline("b", TRI[1], TRI[2]), pline("c", TRI[2], TRI[0])]
    s = metrics.score_objects(gt, sides)
    assert s["closed_as_sides"] == 1 and s["structural"]["recall"] == 1.0 and s["strict"]["recall"] == 0.0


def test_triangle_polygon_iou_and_type_confusion():
    gt = gt_doc([tri("t1", TRI)])
    s = metrics.score_objects(gt, [ppoly("p", TRI, typ="polygon")])
    assert s["strict"]["tp"] == 1
    assert s["geometry_errors"]["polygon_iou"][0] > 0.99
    assert s["type_correct"] == 0 and s["type_confusion"] == {"triangle->polygon": 1}


def test_omittable_ground_truth_is_neither_tp_nor_fp():
    gt = gt_doc([seg("g", (100, 500), (900, 500), typ="grid_line", importance="omittable"), seg("s", (100, 100), (900, 100))])
    s = metrics.score_objects(gt, [pline("x", (100, 500), (900, 500)), pline("y", (100, 100), (900, 100))])
    assert s["gt_counted"] == 1 and s["strict"]["tp"] == 1 and s["strict"]["fp"] == 0 and s["on_omittable"] == 1


def test_points_and_angles_are_not_object_predictions():
    gt = gt_doc([seg("s", (100, 100), (900, 100))])
    preds = [pline("y", (100, 100), (900, 100)), {"id": "p", "type": "point", "geometry": {"position": [5, 5]}},
             {"id": "a", "type": "angle", "geometry": {"vertex": [5, 5]}}]
    assert metrics.score_objects(gt, preds)["strict"]["fp"] == 0


def test_false_positive_on_text_is_attributed():
    gt = gt_doc([seg("s", (100, 100), (900, 100))],
                labels=[{"id": "l", "text": "A", "bbox": [600, 600, 40, 40], "semantic_importance": "essential"}])
    s = metrics.score_objects(gt, [pline("y", (100, 100), (900, 100)), pline("z", (605, 620), (635, 620))])
    assert s["fp_reasons"] == {"on_text": 1}


def test_label_scoring_cer_split_tokens_association_and_braille():
    gt = gt_doc(
        [seg("s", (100, 100), (900, 100))],
        labels=[{"id": "l1", "text": "AB = 5 cm", "bbox": [400, 50, 120, 30], "semantic_importance": "essential",
                 "associated_object": "s"},
                {"id": "l2", "text": "Q", "bbox": [10, 700, 20, 20], "semantic_importance": "essential"}],
    )
    preds = [pline("e1", (100, 100), (900, 100))]
    labels = [{"id": "t1", "text": "AB = 5", "bbox": [400, 50, 70, 30], "associated_element_id": "e1", "braille": "x"},
              {"id": "t2", "text": "cm", "bbox": [480, 50, 40, 30], "associated_element_id": None, "braille": "y"}]
    s = metrics.score_labels(gt, labels, {"l1": "zz", "l2": "q"}, lambda t: t, preds, {"s": ["e1"]})
    assert s["detected"] == 1 and s["detection_recall"] == 0.5
    assert s["char_errors"] == 1  # "Q" missed entirely; split tokens rejoin exactly
    assert s["chars"] == len("ab = 5 cm") + 1
    assert s["association_total"] == 1 and s["association_correct"] == 1
    assert s["quantitative_labels"] == 1 and s["quantitative_preserved"] == 1
    assert s["braille_total"] == 0  # split labels are not scored cell by cell


def test_braille_cells_and_reading_order():
    gt = gt_doc([], labels=[
        {"id": "a", "text": "A", "bbox": [100, 100, 20, 20], "semantic_importance": "essential"},
        {"id": "b", "text": "B", "bbox": [300, 100, 20, 20], "semantic_importance": "essential"},
    ])
    labels = [{"id": "t1", "text": "A", "bbox": [100, 100, 20, 20], "braille": "⠁", "reading_order": 2},
              {"id": "t2", "text": "B", "bbox": [300, 100, 20, 20], "braille": "⠃⠃", "reading_order": 1}]
    s = metrics.score_labels(gt, labels, {"a": "⠁", "b": "⠃"}, lambda t: {"A": "⠁", "B": "⠃"}[t])
    assert s["braille_total"] == 2 and s["braille_exact"] == 1 and s["braille_consistent"] == 1
    assert s["braille_cell_errors"] == 1 and s["braille_cells"] == 2
    assert s["reading_order_pairs"] == 1 and s["reading_order_concordant"] == 0


def test_relationship_recall_and_unsupported_types():
    gt = gt_doc([seg("s1", (100, 100), (900, 100)), seg("s2", (100, 300), (900, 300))],
                relationships=[{"type": "PARALLEL", "source": "s1", "target": "s2"},
                               {"type": "REFLECTION_OF", "source": "s1", "target": "s2"}])
    preds = [pline("a", (100, 100), (900, 100)), pline("b", (100, 300), (900, 300))]
    rels = [{"type": "parallel_lines", "element_ids": ["b", "a"]}, {"type": "perpendicular_lines", "element_ids": ["a", "b"]}]
    s = metrics.score_relationships(gt, preds, rels, {"s1": ["a"], "s2": ["b"]})
    assert s["found"] == 1 and s["recall"] == 1.0
    assert s["gt_unsupported"] == {"REFLECTION_OF": 1}
    assert s["predicted_mappable"] == 2 and s["predicted_mappable_correct"] == 1


def test_svg_structure_checks():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           '<line x1="0" y1="0" x2="50" y2="50"/><line x1="0" y1="0" x2="50" y2="50"/>'
           '<polygon points="0,0 10,10 10,0 0,10"/><circle cx="95" cy="50" r="10"/></svg>')
    s = metrics.score_svg(svg)
    assert s["valid"] and s["duplicates"] == 1 and s["self_intersecting_polygons"] == 1 and s["out_of_bounds"] == 1
    assert metrics.score_svg("<svg") ["valid"] is False
    assert metrics.score_svg(None)["present"] is False


def test_summarize_counts_failed_images():
    rows = [{"score": {"failed": True, "error": "x", "gt_essential": 4}, "timings_ms": {"total": 10}}]
    s = summarize(rows)
    assert s["failed"] == 1 and s["failure_rate"] == 1.0


# ----------------------------------------------------------- line grouping
def test_line_grouping_classifies_edges():
    gt = gt_doc([seg("a", (100, 100), (500, 100)), seg("b", (100, 300), (500, 300)),
                 seg("c", (100, 500), (500, 500)), seg("d", (100, 700), (500, 700))])
    preds = [
        pline("p1", (101, 100), (499, 101)),                                 # a: once
        pline("p2", (100, 300), (290, 300)), pline("p3", (310, 300), (500, 300)),  # b: fragmented
        pline("p4", (100, 500), (500, 500)), pline("p5", (102, 502), (498, 502)),  # c: duplicated
    ]                                                                         # d: missed
    g = metrics.score_line_grouping(gt, preds)
    assert (g["edges"], g["correct"], g["fragmented"], g["duplicated"], g["missed"]) == (4, 1, 1, 1, 1)
    assert g["over_segmentation_rate"] == round(1 / 3, 4)
    assert g["missed_rate"] == 0.25


def test_line_grouping_counts_polygon_sides():
    gt = gt_doc([tri("t", TRI)])
    whole = metrics.score_line_grouping(gt, [ppoly("p", TRI)])
    assert whole["edges"] == 3 and whole["correct"] == 3
    sides = metrics.score_line_grouping(gt, [pline(f"s{k}", TRI[k], TRI[(k + 1) % 3]) for k in range(3)])
    assert sides["correct"] == 3


def test_line_grouping_ignores_omittable_and_crossing_lines():
    gt = gt_doc([seg("g", (0, 50), (1000, 50), typ="grid_line", importance="omittable"),
                 seg("a", (100, 100), (500, 100))])
    g = metrics.score_line_grouping(gt, [pline("x", (300, 0), (300, 400))])
    assert g["edges"] == 1 and g["missed"] == 1


# --------------------------------------------------- semantic preservation
def test_semantic_preservation_counts_unsupported_relations_as_lost():
    gt = gt_doc([seg("a", (100, 100), (500, 100))],
                labels=[{"id": "l1", "text": "A", "semantic_importance": "essential", "associated_object": "a"}],
                relationships=[{"type": "LABELS", "source": "l1", "target": "a"},
                               {"type": "INSIDE", "source": "a", "target": "a"}])
    objects = {"essential_found_structural": 1, "gt_essential": 1, "_found_ids": {"a"}}
    labels = {"per_label": [{"id": "l1", "detected": True, "char_errors": 0}], "exact_match_normalized": 1,
              "gt_essential": 1, "association_correct": 1}
    spr = metrics.semantic_preservation(gt, objects, labels, {"found": 0})
    assert spr["components"]["relationships"] == {"preserved": 0, "total": 1, "rate": 0.0}
    assert spr["preserved"] == 3 and spr["total"] == 4 and spr["rate"] == 0.75


# ------------------------------------------------------------------ QA gate
def test_qa_gate_status_and_unblocked_critical():
    ok_svg = {"present": True, "valid": True, "out_of_bounds": 0}
    warn = {"issues": [{"check": "spacing", "severity": "warning"}]}
    assert metrics.qa_gate({"issues": []}, ok_svg, 0)["status"] == "PASS"
    g = metrics.qa_gate(warn, {**ok_svg, "out_of_bounds": 2}, 1)
    assert g["status"] == "WARNING"
    assert g["critical_unblocked"] == ["essential_removed_by_simplification", "out_of_bounds"]
    blocked = metrics.qa_gate({"issues": [{"check": "braille_collision", "severity": "error"}]}, ok_svg, 0)
    assert blocked["status"] == "BLOCKED" and blocked["critical_unblocked"] == []
    assert blocked["critical_present"] == ["braille_collision"]


def test_relationship_by_type_precision_counts():
    gt = gt_doc([seg("a", (100, 100), (500, 100)), seg("b", (100, 300), (500, 300))],
                relationships=[{"type": "PARALLEL", "source": "a", "target": "b"}])
    preds = [pline("p1", (100, 100), (500, 100)), pline("p2", (100, 300), (500, 300))]
    rels = [{"type": "parallel_lines", "element_ids": ["p1", "p2"]},
            {"type": "perpendicular_lines", "element_ids": ["p1", "p2"]}]
    s = metrics.score_relationships(gt, preds, rels, {"a": ["p1"], "b": ["p2"]})
    assert s["by_type"]["PARALLEL"]["predicted_correct"] == 1
    assert s["by_type"]["PERPENDICULAR"]["predicted_correct"] == 0
    assert s["by_type"]["PERPENDICULAR"]["predicted"] == 1
