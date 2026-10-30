# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_compile_and_are_committed_without_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), (
            f"{path.name} cell {index} has committed outputs"
        )
        # IPython magics are not Python syntax; the rest of the cell is.
        source = "".join(
            line for line in cell["source"] if not line.lstrip().startswith(("%", "!"))
        )
        compile(source, f"{path.name}:cell-{index}", "exec")


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_executes_top_to_bottom_in_a_clean_kernel(path):
    """Run every cell in order in a fresh kernel; any exception fails the test.

    The notebook file is not modified. Split MNIST cells run if ``data/mnist`` exists and
    print a skip message otherwise, so this works on a fresh checkout.
    """
    nbformat = pytest.importorskip("nbformat")
    preprocessors = pytest.importorskip("nbconvert.preprocessors")
    notebook = nbformat.read(path, as_version=4)
    executor = preprocessors.ExecutePreprocessor(
        timeout=3600, kernel_name="python3", allow_errors=False
    )
    executor.preprocess(notebook, {"metadata": {"path": str(path.parent)}})
    # Part I of the main notebook must have produced its tables, not silently skipped.
    if path.name.startswith("02_"):
        printed = "".join(
            output.get("text", "")
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.outputs
        )
        assert "fixed_capacity" in printed and "ordinary_no_replay" in printed


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_report_sample_standard_deviations(path):
    """Results are documented as sample std (ddof=1); NumPy's default is the population std."""
    import re

    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for call in re.finditer(r"\bstd\(([^()]*(?:\([^()]*\)[^()]*)*)\)", source):
            assert "ddof=1" in call.group(1), (path.name, index, call.group(0))
