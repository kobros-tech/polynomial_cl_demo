# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.multiclass import fit_task


def test_class_incremental_training_does_not_update_future_class_weights():
    features = np.array([[1.0, 0.5], [-0.5, 1.0], [0.2, -1.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    weights = np.zeros((2, 10), dtype=np.float32)

    trained = fit_task(
        weights,
        features,
        labels,
        np.empty((0, 2), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        epochs=2,
        learning_rate=0.1,
        batch_size=3,
        seed=5,
        eligible_classes=np.array([0, 1]),
    )

    assert np.any(trained[:, :2] != 0)
    np.testing.assert_array_equal(trained[:, 2:], 0.0)


def test_class_incremental_training_rejects_labels_outside_eligible_classes():
    with pytest.raises(ValueError, match="belong to eligible_classes"):
        fit_task(
            np.zeros((2, 4), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 2], dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            epochs=1,
            learning_rate=0.1,
            batch_size=2,
            seed=0,
            eligible_classes=np.array([0, 1]),
        )
