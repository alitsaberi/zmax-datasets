import numpy as np

from zmax_datasets.datasets.utils import squeeze_ids


def test_squeeze_ids():
    initials = np.array([0, 30, 60, 90, 120])
    durations = np.array([30, 30, 30, 30, 30])
    annotations = np.array([0, 0, 1, 1, 0])

    expected_initials = np.array([0, 60, 120])
    expected_durations = np.array([60, 60, 30])
    expected_annotations = np.array([0, 1, 0])

    squeezed_initials, squeezed_durations, squeezed_annotations = squeeze_ids(
        initials, durations, annotations
    )

    np.testing.assert_array_equal(squeezed_initials, expected_initials)
    np.testing.assert_array_equal(squeezed_durations, expected_durations)
    np.testing.assert_array_equal(squeezed_annotations, expected_annotations)
