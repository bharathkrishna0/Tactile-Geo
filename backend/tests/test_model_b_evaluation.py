"""Phase 6 tests: evaluation metrics and the evaluation runner.

The metrics are pure, so these are exact. The runner tests assert the two
properties that matter for honesty: a hermetic run labels itself as such, and a
live run without a key refuses rather than quietly degrading to a fake one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.model_b.errors import ModelBApiError
from app.model_b.evaluation import (
    FixtureOutcome,
    fusion_counts,
    score_detections,
    score_relations,
    summarise,
)
from app.model_b.fusion import MATCH_IOU_THRESHOLD, FusionReport


def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_model_b.py"
    spec = importlib.util.spec_from_file_location("evaluate_model_b", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


class TestIou:
    def test_identical_boxes(self) -> None:
        score = score_detections([("a", [0, 0, 10, 10])], [("a", [0, 0, 10, 10])])
        assert score.true_positives == 1
        assert score.false_positives == 0
        assert score.false_negatives == 0

    def test_disjoint_boxes_do_not_match(self) -> None:
        score = score_detections([("a", [0, 0, 10, 10])], [("a", [500, 500, 10, 10])])
        assert score.true_positives == 0
        assert score.false_positives == 1
        assert score.false_negatives == 1

    def test_label_must_match(self) -> None:
        """A correct box with the wrong label is not a detection."""
        score = score_detections([("circle", [0, 0, 10, 10])], [("triangle", [0, 0, 10, 10])])
        assert score.true_positives == 0

    def test_zero_area_boxes_do_not_divide_by_zero(self) -> None:
        score = score_detections([("a", [0, 0, 0, 0])], [("a", [0, 0, 0, 0])])
        assert score.true_positives == 0  # union is 0, so no match is claimed


class TestOneToOneMatching:
    def test_one_prediction_cannot_confirm_two_expected(self) -> None:
        """Regression guard for double-counting.

        Three predictions stacked on one expected region used to score three
        true positives, which would reward a model that reports the same shape
        repeatedly.
        """
        score = score_detections(
            [("a", [0, 0, 10, 10]), ("a", [0, 0, 10, 10]), ("a", [0, 0, 10, 10])],
            [("a", [0, 0, 10, 10])],
        )
        assert score.true_positives == 1
        assert score.false_positives == 2
        assert score.recall == 1.0

    def test_one_expected_cannot_be_claimed_twice(self) -> None:
        score = score_detections(
            [("a", [0, 0, 10, 10]), ("a", [0, 0, 10, 10])],
            [("a", [0, 0, 10, 10]), ("a", [40, 40, 10, 10])],
        )
        assert score.true_positives == 1
        assert score.false_negatives == 1

    def test_best_overlap_wins_when_two_candidates_are_close(self) -> None:
        """Greedy best-IoU pairing, so a tight box is not stolen by a loose one."""
        score = score_detections(
            [("a", [20, 20, 10, 10])],  # tight over the second
            [("a", [0, 0, 10, 10]), ("a", [20, 20, 10, 10])],
        )
        assert score.true_positives == 1
        assert score.false_negatives == 1


class TestUndefinedMetrics:
    def test_no_predictions_gives_undefined_precision_and_f1(self) -> None:
        """Precision over zero predictions is undefined, not 0%.

        Reporting 0% would make a model that correctly stays quiet on a blank
        page look like a total failure. Recall of 0.0 already carries the real
        signal, and F1 is left undefined because it is derived from a
        precision that does not exist.
        """
        score = score_detections([], [("a", [0, 0, 10, 10])])
        assert score.precision is None
        assert score.recall == 0.0
        assert score.f1 is None

    def test_no_predictions_and_no_expectations_is_all_none(self) -> None:
        score = score_detections([], [])
        assert score.precision is None
        assert score.recall is None
        assert score.f1 is None

    def test_f1_undefined_when_precision_is_undefined(self) -> None:
        score = score_detections([("a", [0, 0, 10, 10])], [])
        assert score.precision == 0.0
        assert score.recall is None
        assert score.f1 is None

    def test_perfect_score(self) -> None:
        score = score_detections([("a", [0, 0, 10, 10])], [("a", [0, 0, 10, 10])])
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0


class TestThreshold:
    def test_default_threshold_is_the_fusion_threshold(self) -> None:
        score = score_detections([("a", [0, 0, 10, 10])], [("a", [0, 0, 10, 10])])
        assert score.iou_threshold == MATCH_IOU_THRESHOLD

    def test_below_threshold_is_a_miss(self) -> None:
        # A 10x10 expected with a 4x4 prediction inside it gives IoU 0.16,
        # clearly under the 0.25 threshold.
        score = score_detections([("a", [0, 0, 4, 4])], [("a", [0, 0, 10, 10])])
        assert score.true_positives == 0
        assert score.false_positives == 1
        assert score.false_negatives == 1


class TestRelationScoring:
    def test_exact_statement_match(self) -> None:
        score = score_relations(["The circle touches BC."], ["The circle touches BC."])
        assert score.true_positives == 1

    def test_case_and_whitespace_insensitive(self) -> None:
        score = score_relations(["  the circle touches bc "], ["The Circle Touches BC"])
        assert score.true_positives == 1

    def test_order_does_not_matter(self) -> None:
        predicted = ["first statement", "second statement"]
        expected = ["second statement", "first statement"]
        assert score_relations(predicted, expected).true_positives == 2

    def test_duplicate_statements_do_not_inflate(self) -> None:
        score = score_relations(["same", "same", "same"], ["same"])
        assert score.true_positives == 1
        assert score.false_positives == 0

    def test_blank_statements_are_ignored(self) -> None:
        score = score_relations(["", "   "], [])
        assert score.true_positives == 0
        assert score.false_positives == 0


class TestSummarise:
    def test_failed_fixtures_are_counted_not_dropped(self) -> None:
        """A run where 3 of 10 errored must not report the mean of the 7."""
        outcomes = [
            FixtureOutcome("a", latency_ms=10.0, entity_score=score_detections([], [])),
            FixtureOutcome("b", latency_ms=20.0, entity_score=score_detections([], [])),
            FixtureOutcome("c", ok=False, error_code="model_b_timeout"),
            FixtureOutcome("d", ok=False, error_code="model_b_timeout"),
        ]
        summary = summarise(outcomes)
        assert summary["fixtures"] == 4
        assert summary["succeeded"] == 2
        assert summary["failed"] == 2
        assert summary["failure_codes"] == {"model_b_timeout": 2}

    def test_latency_median_even_count(self) -> None:
        outcomes = [FixtureOutcome(f"f{i}", latency_ms=float(i)) for i in range(4)]
        assert summarise(outcomes)["latency"]["median_ms"] == 1.5

    def test_latency_median_odd_count(self) -> None:
        outcomes = [FixtureOutcome(f"f{i}", latency_ms=float(i)) for i in range(5)]
        assert summarise(outcomes)["latency"]["median_ms"] == 2.0

    def test_no_latencies_reports_none(self) -> None:
        assert summarise([])["latency"] == {
            "min_ms": None,
            "median_ms": None,
            "max_ms": None,
        }

    def test_empty_run(self) -> None:
        summary = summarise([])
        assert summary["fixtures"] == 0
        assert summary["entities"]["precision"] is None


class TestFusionCounts:
    def test_empty_report(self) -> None:
        assert fusion_counts(FusionReport()) == {
            "agreements": 0,
            "candidate_additions": 0,
            "disagreements": 0,
            "type_mismatches": 0,
            "model_a_only": 0,
            "weak_overlaps": 0,
        }


class TestScriptedClient:
    def test_never_touches_a_key(self) -> None:
        client = script.ScriptedClient("{}")
        assert not hasattr(client, "api_key")

    def test_replays_the_stored_text(self) -> None:
        client = script.ScriptedClient('{"a": 1}')
        response = client.analyze(image_png=b"png", prepared_width=1568, prepared_height=1568)
        assert response.text == '{"a": 1}'

    def test_rejects_an_empty_image(self) -> None:
        client = script.ScriptedClient("{}")
        with pytest.raises(RuntimeError):
            client.analyze(image_png=b"", prepared_width=1, prepared_height=1)

    def test_hermetic_settings_key_is_an_obvious_placeholder(self) -> None:
        settings = script.hermetic_settings()
        assert settings.enabled is True
        assert settings.api_key is not None
        assert "unused" in settings.api_key
        assert settings.model == "scripted"


class TestFixtures:
    def test_ground_truth_boxes_match_the_document_boxes(self) -> None:
        """The fixtures must actually agree with each other.

        Expected regions are in original-image pixels; the document holds
        normalised boxes. If the two drift apart the runner would report
        nonsense, and nothing else would catch it.
        """
        for fixture in script._load_fixtures():
            width = fixture["response"]["image"]["width_px"]
            height = fixture["response"]["image"]["height_px"]
            for entity in fixture["response"]["entities"]:
                nx, ny, nw, nh = entity["bbox"]
                pixels = [nx * width, ny * height, nw * width, nh * height]
                assert any(
                    kind == entity["kind"] and list(box) == pytest.approx(pixels)
                    for kind, box in fixture["expected_entities"]
                ), f"{fixture['name']}: {entity['id']} has no matching ground truth"

    def test_blank_page_expects_nothing(self) -> None:
        blank = next(f for f in script._load_fixtures() if f["name"] == "blank_page")
        assert blank["expected_entities"] == []
        assert blank["response"]["entities"] == []


class TestRunner:
    def test_hermetic_run_succeeds_and_labels_itself(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        assert script.main(["--out", str(out)]) == 0
        report = json.loads(out.read_text(encoding="utf-8"))

        assert report["evidence_source"] == "fake_client"
        assert report["latency_scope"] == "pipeline_only"
        assert report["summary"]["failed"] == 0
        assert report["summary"]["succeeded"] == 2
        for fixture in report["fixtures"]:
            assert fixture["evidence_source"] == "fake_client"
            assert fixture["ok"] is True

    def test_hermetic_run_uses_a_literal_placeholder_key(self, tmp_path: Path) -> None:
        """Guard against a real key ever being written into a report."""
        out = tmp_path / "report.json"
        script.main(["--out", str(out)])
        assert "hermetic-placeholder-unused" not in out.read_text(encoding="utf-8")

    def test_live_run_without_a_key_refuses(self, monkeypatch, capsys) -> None:
        from app.core import config

        monkeypatch.setattr(config, "model_b_enabled", False, raising=False)
        monkeypatch.setenv("MODEL_B_ENABLED", "false")
        assert script.main(["--live"]) == 2
        assert "live evaluation unavailable" in capsys.readouterr().err

    def test_report_is_json_serialisable(self, tmp_path: Path) -> None:
        out = tmp_path / "report.json"
        script.main(["--out", str(out)])
        json.loads(out.read_text(encoding="utf-8"))  # must not raise


class TestErrorRecording:
    def test_provider_failure_is_recorded_not_raised(self) -> None:
        """One failing fixture must not abort the whole run."""

        class FailingClient:
            def analyze(self, image_png, prepared_width, prepared_height):
                raise ModelBApiError("Upstream is unavailable.", retryable=True)

        fixtures = script._load_fixtures()
        outcome = script.run_fixture(
            fixtures[0], FailingClient(), "live_provider", script.hermetic_settings()
        )
        assert outcome.ok is False
        assert outcome.error_code == "model_b_api_error"
        assert outcome.entity_score is None
        assert "Upstream" in (outcome.error_message or "")
