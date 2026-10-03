"""Integrity checks for the committed math100 dataset."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import pytest

DATASET = Path(__file__).resolve().parents[1] / "dataset"
META = json.loads((DATASET / "metadata" / "dataset.json").read_text())
ANNOTATIONS = {p.stem: json.loads(p.read_text()) for p in sorted((DATASET / "annotations").glob("*.json"))}
CATEGORIES = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}


def test_counts_tiers_and_categories():
    assert len(ANNOTATIONS) == 100 == META["count"]
    assert Counter(a["difficulty"] for a in ANNOTATIONS.values()) == {
        "easy": 20, "moderate": 20, "hard": 20, "very_hard": 20, "adversarial": 20}
    assert {a["category"][0] for a in ANNOTATIONS.values()} == CATEGORIES


def test_splits_are_disjoint_and_cover_everything():
    splits = {s: (DATASET / "splits" / f"{s}.txt").read_text().split() for s in ("train", "validation", "test")}
    assert [len(splits[s]) for s in ("train", "validation", "test")] == [70, 15, 15]
    ids = [i for v in splits.values() for i in v]
    assert len(ids) == len(set(ids)) == 100 and set(ids) == set(ANNOTATIONS)
    test_tiers = Counter(ANNOTATIONS[i]["difficulty"] for i in splits["test"])
    assert set(test_tiers.values()) == {3}


@pytest.mark.parametrize("image_id", sorted(ANNOTATIONS))
def test_annotation_is_exact_and_in_bounds(image_id):
    a = ANNOTATIONS[image_id]
    assert a["schema_version"] == META["schema_version"]
    assert a["annotation"]["method"] == "render_time_exact" and a["annotation"]["machine_assisted"] is False
    image = cv2.imread(str(DATASET / a["file"]))
    mask = cv2.imread(str(DATASET / a["mask_file"]), cv2.IMREAD_GRAYSCALE)
    assert image.shape[:2] == mask.shape[:2] == (a["height"], a["width"])
    meta = next(m for m in META["images"] if m["image_id"] == image_id)
    assert hashlib.sha256((DATASET / a["file"]).read_bytes()).hexdigest() == meta["sha256_image"]
    ids = [o["id"] for o in a["objects"]] + [p["id"] for p in a["points"]] + [lab["id"] for lab in a["labels"]]
    assert len(ids) == len(set(ids)), "ids must be unique"
    known = set(ids)
    for r in a["relationships"]:
        assert r["source"] in known and r["target"] in known
    for lab in a["labels"]:
        assert lab["text"].strip()
        if lab.get("associated_object"):
            assert lab["associated_object"] in known
    assert any(o["semantic_importance"] == "essential" for o in a["objects"]) or a["category"].startswith("K")
    assert (mask > 0).any()
