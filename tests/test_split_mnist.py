# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import gzip
import struct

import numpy as np

from experiments.split_mnist_utils import read_idx


def test_read_idx_images_and_labels(tmp_path):
    images = tmp_path / "images.gz"
    raw_images = struct.pack(">IIII", 2051, 2, 2, 2) + bytes([0, 1, 2, 3, 4, 5, 6, 7])
    with gzip.open(images, "wb") as f:
        f.write(raw_images)
    got = read_idx(images, image=True)
    assert got.shape == (2, 4)
    np.testing.assert_array_equal(got[1], [4, 5, 6, 7])

    labels = tmp_path / "labels.gz"
    with gzip.open(labels, "wb") as f:
        f.write(struct.pack(">II", 2049, 3) + bytes([1, 4, 9]))
    np.testing.assert_array_equal(read_idx(labels, image=False), [1, 4, 9])


def test_read_idx_rejects_truncated_image_payload(tmp_path):
    path = tmp_path / "truncated-images.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, 2, 2, 2) + bytes([0, 1, 2]))
    import pytest

    with pytest.raises(ValueError, match="invalid image IDX length"):
        read_idx(path, image=True)


def test_read_idx_rejects_invalid_label_magic(tmp_path):
    path = tmp_path / "bad-labels.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", 1234, 1) + bytes([0]))
    import pytest

    with pytest.raises(ValueError, match="invalid label IDX magic"):
        read_idx(path, image=False)
