# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

"""Compatibility imports for the package-owned MNIST loader."""

from polynomial_cl.mnist import BASE_URL, FILES, fetch, load_mnist, read_idx

__all__ = ["BASE_URL", "FILES", "fetch", "load_mnist", "read_idx"]
