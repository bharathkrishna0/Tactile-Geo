from evaluation.run_pages import figure_annotation, match_figures


def _page():
    return {
        "width": 1000, "height": 1400,
        "objects": [
            {"id": "circle_1", "type": "circle", "bbox": [100, 200, 60, 60],
             "geometry": {"kind": "circle", "center": [130, 230], "radius": 30},
             "outline": [[100, 230], [160, 230]]},
            {"id": "segment_1", "type": "line_segment", "bbox": [100, 300, 50, 0],
             "geometry": {"kind": "segment", "points": [[100, 300], [150, 300]]}},
            {"id": "segment_2", "type": "line_segment", "bbox": [700, 900, 50, 0],
             "geometry": {"kind": "segment", "points": [[700, 900], [750, 900]]}},
        ],
        "points": [{"id": "point_1", "x": 100, "y": 300}, {"id": "point_2", "x": 700, "y": 900}],
        "labels": [{"id": "label_1", "bbox": [90, 310, 10, 12], "text": "A"}],
        "relationships": [
            {"type": "ENDPOINT_OF", "source": "point_1", "target": "segment_1"},
            {"type": "ENDPOINT_OF", "source": "point_2", "target": "segment_2"},
        ],
        "dimensions": [], "omittable_ids": ["segment_2"],
        "figures": [{"id": "figure_1", "box": [80, 180, 200, 340],
                     "object_ids": ["circle_1", "segment_1"], "point_ids": ["point_1"], "label_ids": ["label_1"]}],
    }


def test_figure_annotation_keeps_only_the_figure_and_moves_every_coordinate():
    page = _page()
    sub = figure_annotation(page, page["figures"][0], 80, 180)
    assert [o["id"] for o in sub["objects"]] == ["circle_1", "segment_1"]
    circle, segment = sub["objects"]
    assert circle["bbox"][:2] == [20, 20] and circle["geometry"]["center"] == [50, 50]
    assert circle["outline"] == [[20, 50], [80, 50]]
    assert segment["geometry"]["points"] == [[20, 120], [70, 120]]
    assert (sub["points"][0]["x"], sub["points"][0]["y"]) == (20, 120)
    assert sub["labels"][0]["bbox"][:2] == [10, 130]
    assert sub["relationships"] == [{"type": "ENDPOINT_OF", "source": "point_1", "target": "segment_1"}]
    assert sub["omittable_ids"] == [] and "figures" not in sub
    assert page["objects"][0]["geometry"]["center"] == [130, 230]


def test_match_figures_pairs_each_box_once_by_overlap():
    gt = [{"bbox": [0, 0, 100, 100]}, {"bbox": [200, 0, 100, 100]}]
    found = [{"box": [205, 5, 300, 100]}, {"box": [0, 0, 90, 100]}, {"box": [500, 500, 600, 600]}]
    assert [(i, j) for i, j, _ in match_figures(gt, found)] == [(0, 1), (1, 0)]
