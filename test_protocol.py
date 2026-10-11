import numpy as np
import pytest

from protocol import (
    GroundTruth, align_predictions, box_iou, holm_adjust,
    paired_statistics, trackeval_data, validate_split,
)


def test_failed_frames_and_entire_failed_targets_remain_in_denominator():
    gt = [
        GroundTruth("s1", "a", 1, (0, 0, 10, 10), "one.jpg"),
        GroundTruth("s1", "a", 2, (0, 0, 10, 10), "two.jpg"),
        GroundTruth("s2", "b", 1, (0, 0, 10, 10), "three.jpg"),
    ]
    rows = align_predictions(gt, {gt[0].key: (0, 0, 10, 10)})
    assert len(rows) == 3
    assert sum(r["bbox_iou"] for r in rows) / 3 == pytest.approx(1 / 3)
    assert sum(r["success_05"] for r in rows) == 1
    assert rows[2]["status"] == "missing_prediction"
    data = trackeval_data(rows)
    assert data["num_gt_dets"] == 3
    assert data["num_tracker_dets"] == 1
    assert data["similarity_scores"][1].shape == (1, 0)


def test_missing_predictions_have_zero_not_infinite_center_precision():
    rows = align_predictions([GroundTruth("s", "a", 1, (1, 2, 3, 4), "")], {})
    assert rows[0]["precision_20"] == 0
    assert rows[0]["center_error_px"] == ""
    assert trackeval_data(rows)["num_tracker_ids"] == 0


def test_duplicate_gt_and_extraneous_predictions_are_errors():
    gt = GroundTruth("s", "a", 1, (0, 0, 10, 10), "")
    with pytest.raises(ValueError, match="unique"):
        align_predictions([gt, gt], {})
    with pytest.raises(ValueError, match="outside"):
        align_predictions([gt], {("other", "a", 1): (0, 0, 10, 10)})


def test_standard_iou_does_not_reward_an_oversized_covering_box():
    assert box_iou((0, 0, 10, 10), (-5, -5, 20, 20)) == pytest.approx(0.25)
    assert box_iou((0, 0, 10, 10), None) == 0


def test_paired_statistics_align_names_not_list_positions():
    result = paired_statistics({"b": 0.4, "a": 0.8}, {"a": 0.7, "b": 0.2})
    assert result["mean_paired_difference"] == pytest.approx(0.15)
    assert result["groups"] == ["a", "b"]
    assert result["ci95_low"] == pytest.approx(0.1)
    assert result["ci95_high"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical"):
        paired_statistics({"a": 1, "b": 2}, {"a": 1, "c": 2})


def test_holm_handles_order_and_bounds():
    np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    assert holm_adjust([1.0, 1.0]) == [1.0, 1.0]
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_identical_group_results_are_reported_as_nonsignificant():
    result = paired_statistics({"a": 0.2, "b": 0.3}, {"a": 0.2, "b": 0.3})
    assert result["p_two_sided"] == 1
    assert result["ci95_low"] == result["ci95_high"] == 0


def test_all_learning_and_selection_stages_require_disjoint_sequences():
    validate_split(["EV1", "EV2"], ["EV3"], ["IV1"])
    with pytest.raises(ValueError, match="leakage"):
        validate_split(["EV1", "IV1"], ["EV3"], ["IV1"])
    with pytest.raises(ValueError, match="nonempty"):
        validate_split(["EV1"], [], ["IV1"])


def test_ground_truth_initialization_is_not_an_accuracy_observation():
    from audit_results import summarize_tracking
    gt = [
        GroundTruth("s1", "a", 1, (0, 0, 10, 10), "", True),
        GroundTruth("s1", "a", 2, (0, 0, 10, 10), ""),
        GroundTruth("s2", "a", 1, (0, 0, 10, 10), "", True),
        GroundTruth("s2", "a", 2, (0, 0, 10, 10), ""),
    ]
    aligned = align_predictions(gt, {r.key: r.box for r in gt if r.is_initialization})
    summary, _, _ = summarize_tracking(aligned)
    assert summary["n_gt_frames"] == 2
    assert summary["n_missing_predictions"] == 2
    assert summary["bbox_iou"]["mean"] == 0
    assert summary["pooled_official_trackeval"]["HOTA"] == 0


def test_official_hota_counts_entire_failed_track_as_false_negatives():
    from audit_results import HOTA
    rows = align_predictions([
        GroundTruth("s", "a", 1, (0, 0, 10, 10), ""),
        GroundTruth("s", "a", 2, (0, 0, 10, 10), ""),
    ], {})
    result = HOTA().eval_sequence(trackeval_data(rows))
    assert np.all(result["HOTA_FN"] == 2)
    assert np.all(result["HOTA_TP"] == 0)
