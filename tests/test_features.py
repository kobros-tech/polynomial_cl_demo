# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from polynomial_cl.class_incremental.features import (
    expand_weights,
)
from polynomial_cl.class_incremental.features import (
    monomial_exponents as class_incremental_monomial_exponents,
)
from polynomial_cl.features import (
    PolynomialFeatures,
    append_zero_coefficients,
    feature_exponents,
    feature_names,
    monomial_exponents,
    n_features,
    polynomial_features,
)
from polynomial_cl.model import PolynomialLogisticModel


def test_names_degree_1_2_3():
    assert feature_names(1) == ["1", "x1", "x2"]
    assert feature_names(2) == ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
    assert feature_names(3)[6:] == ["x1^3", "x1^2*x2", "x1*x2^2", "x2^3"]


def test_ordering_is_deterministic_and_prefix_stable():
    assert feature_names(6) == feature_names(6)
    for d in range(9):
        assert feature_names(d + 1)[: n_features(d)] == feature_names(d)


@pytest.mark.parametrize("d", range(13))
def test_feature_count(d):
    assert n_features(d) == (d + 1) * (d + 2) // 2
    assert len(feature_names(d)) == n_features(d)
    assert polynomial_features(np.zeros((3, 2)), d).shape == (3, n_features(d))


def test_known_values():
    x = np.array([[2.0, 3.0]])
    np.testing.assert_array_equal(polynomial_features(x, 2)[0], [1, 2, 3, 4, 6, 9])
    np.testing.assert_array_equal(
        polynomial_features(x, 3)[0], [1, 2, 3, 4, 6, 9, 8, 12, 18, 27]
    )
    np.testing.assert_array_equal(polynomial_features([-1.0, 0.5], 1)[0], [1, -1, 0.5])


def test_wrapper_exposes_metadata():
    f = PolynomialFeatures(4)
    assert f.degree == 4 and f.n_features == 15 and f.names == feature_names(4)


@pytest.mark.parametrize(
    "degree, error",
    [
        (2.9, TypeError),
        (True, TypeError),
        (np.float64(3), TypeError),
        (np.bool_(True), TypeError),
        (-1, ValueError),
    ],
)
def test_invalid_degree_is_rejected_consistently(degree, error):
    with pytest.raises(error, match="degree"):
        PolynomialFeatures(degree)
    with pytest.raises(error, match="degree"):
        feature_names(degree)
    with pytest.raises(error, match="degree"):
        n_features(degree)
    with pytest.raises(error, match="degree"):
        polynomial_features([[0.0, 0.0]], degree)


def test_numpy_integer_degree_is_accepted():
    assert PolynomialFeatures(np.int64(2)).degree == 2


# ----------------------------------------- shared by binary and class-incremental code
def test_binary_exponents_are_the_two_variable_case_of_the_shared_enumeration():
    assert feature_exponents(3) == [
        (0, 0),
        (1, 0), (0, 1),
        (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3),
    ]  # fmt: skip
    for degree in range(11):
        shared = monomial_exponents(2, degree)
        assert feature_exponents(degree) == [tuple(map(int, row)) for row in shared]
        assert len(shared) == n_features(degree)


def test_class_incremental_features_use_the_same_enumeration():
    assert class_incremental_monomial_exponents is monomial_exponents
    for n_variables in (1, 2, 5):
        for degree in range(4):
            small = monomial_exponents(n_variables, degree)
            np.testing.assert_array_equal(
                monomial_exponents(n_variables, degree + 1)[: len(small)], small
            )


def test_monomial_enumeration_validates_arguments():
    with pytest.raises(ValueError):
        monomial_exponents(0, 2)
    with pytest.raises(ValueError):
        monomial_exponents(2, -1)
    with pytest.raises(TypeError):
        monomial_exponents(2, 1.5)


@pytest.mark.parametrize("shape", [(3,), (3, 4)])
def test_append_zero_coefficients_keeps_old_rows_and_returns_a_copy(shape):
    old = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    new = append_zero_coefficients(old, 7)
    assert new.shape == (7, *shape[1:]) and new.dtype == old.dtype
    np.testing.assert_array_equal(new[:3], old)
    assert not new[3:].any()
    same_size = append_zero_coefficients(old, 3)
    np.testing.assert_array_equal(same_size, old)
    assert not np.shares_memory(same_size, old)
    with pytest.raises(ValueError):
        append_zero_coefficients(old, 2)


def test_binary_model_and_softmax_head_expand_through_the_same_helper():
    model = PolynomialLogisticModel(2, w=np.arange(1.0, 7.0))
    bigger = model.expanded(4)
    np.testing.assert_array_equal(bigger.w[:6], model.w)
    assert not bigger.w[6:].any() and bigger.w.dtype == model.w.dtype
    bigger.w[0] = -1.0  # the expanded model must not alias the original
    assert model.w[0] == 1.0

    head = np.ones((17, 10), dtype=np.float32)
    assert expand_weights(head, 17) is head
    np.testing.assert_array_equal(
        expand_weights(head, 153), append_zero_coefficients(head, 153)
    )
