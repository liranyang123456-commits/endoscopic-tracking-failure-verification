import numpy as np
import pytest

from evaluate_verification import area_under_risk_coverage, grouped_metrics, metrics, risk_values


def observations(labels, sequences=None):
    sequences = sequences or ["s1"] * len(labels)
    return [
        {
            "sequence": sequence, "case": "case1", "failure": label, "box_iou": 1 - label,
            "contour_50": str(.2 * label), "contour_68": str(.3 * label),
            "contour_80": str(.4 * label), "rank_risk": str(.5 * label),
            "color_risk": str(.6 * label), "native_risk": str(.7 * label),
        }
        for label, sequence in zip(labels, sequences)
    ]


def test_perfect_failure_ranking_is_detected_without_changing_boxes():
    rows = observations([0, 0, 1, 1])
    risks = risk_values(rows, .8, [.35, .25, .2, .2])
    result = metrics(rows, risks, .3)
    assert result["failure_auroc"] == 1
    assert result["failure_average_precision"] == 1
    assert result["sensitivity"] == 1
    assert result["specificity"] == 1
    assert result["box_iou_unchanged"] == .5


def test_single_class_sequence_is_kept_but_auc_is_not_invented():
    rows = observations([0, 0, 0])
    result = metrics(rows, np.asarray([.1, .2, .3]), .25)
    assert result["failure_auroc"] is None
    assert result["failure_average_precision"] is None
    assert result["sensitivity"] is None
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["n_frames"] == 3
    assert result["risk_coverage_area"] == 0


def test_tied_risks_do_not_use_ground_truth_to_order_acceptance():
    labels = np.asarray([1, 0, 1, 0])
    risks = np.ones(4)
    assert area_under_risk_coverage(labels, risks) == .5
    assert area_under_risk_coverage(labels[::-1], risks[::-1]) == .5
    assert metrics(observations(labels), risks, 1)["alarm_rate"] == 0


def test_complete_groups_are_summarized_by_identity():
    rows = observations([0, 1, 1, 1], ["a", "a", "b", "b"])
    groups = grouped_metrics(rows, np.asarray([.1, .9, .7, .8]), .5)
    assert groups["a"]["failure_auroc"] == 1
    assert groups["b"]["failure_auroc"] is None
    assert sum(group["n_frames"] for group in groups.values()) == 4


def test_invalid_risk_weights_are_not_silently_normalized():
    with pytest.raises(ValueError, match="sum to one"):
        risk_values(observations([0, 1]), .8, [1, 1, 1, 1])
