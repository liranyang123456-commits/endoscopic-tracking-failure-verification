import cv2
import numpy as np
import pytest

from contour_verifier import (
    ContourVerifier, contour_map, crop_resize, ordinal_image,
    symmetric_trimmed_distance,
)


def test_ranks_are_exactly_invariant_to_strictly_increasing_intensity_maps():
    image = np.arange(64 * 64, dtype=float).reshape(64, 64) / 4096
    np.testing.assert_array_equal(ordinal_image(image), ordinal_image(image ** 2 + 3))
    np.testing.assert_array_equal(
        contour_map(ordinal_image(image), .68),
        contour_map(ordinal_image(image ** 2 + 3), .68),
    )


def test_midrank_ties_are_equal_and_constant_images_are_finite():
    rank = ordinal_image(np.full((64, 64), 7))
    assert np.isfinite(rank).all()
    assert np.unique(rank).size == 1
    assert not contour_map(rank, .68).any()


def test_symmetric_distance_is_zero_for_matching_nonempty_contours():
    image = np.zeros((64, 64), np.uint8)
    cv2.rectangle(image, (10, 12), (40, 45), 1, 1)
    assert symmetric_trimmed_distance(image, image) == 0
    assert symmetric_trimmed_distance(image, np.zeros_like(image)) == 1


def test_verification_never_mutates_or_replaces_proposal():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    box = (10, 12, 64, 64)
    before = image.copy()
    verifier = ContourVerifier(image, box)
    result = verifier.measure(image, box, .9)
    assert result.rank_disagreement < .02
    assert result.color_disagreement < .001
    assert result.native_confidence_risk == pytest.approx(.1)
    np.testing.assert_array_equal(image, before)
    assert box == (10, 12, 64, 64)


def test_invalid_images_and_outside_rois_raise_explicitly():
    with pytest.raises(ValueError):
        ordinal_image(np.zeros((1, 1, 3)))
    with pytest.raises(ValueError):
        crop_resize(np.zeros((64, 64, 3), np.uint8), (100, 100, 10, 10))
    with pytest.raises(ValueError):
        contour_map(np.zeros((64, 64), np.float32), 1)
