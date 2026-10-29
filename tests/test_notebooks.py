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
