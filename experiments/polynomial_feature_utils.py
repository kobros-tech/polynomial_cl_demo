# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Compatibility imports for multiclass experiment feature helpers.

The implementation lives in polynomial_cl.multiclass_features so notebooks and
experiment runners use the package rather than importing another experiment module.
"""

from polynomial_cl.multiclass_features import (
    MAX_DEGREE,
    N_COMPONENTS,
    expand_weights,
    make_projection,
    monomial_exponents,
    n_polynomial_features,
    polynomial_features_projected,
    project_images,
)

CONDITIONS = ("fixed_linear", "expanding", "fixed_capacity")
