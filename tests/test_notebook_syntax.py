# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import json
from pathlib import Path


def test_continual_learning_notebook_code_cells_are_syntactically_valid():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "02_continual_learning.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    code_cells = [
        (index, cell["source"])
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]
    assert code_cells

    for index, source in code_cells:
        # IPython directives are not Python syntax; the remaining cell body is.
        python_source = "\n".join(
            line for line in source
            if not line.lstrip().startswith(("%", "!"))
        )
        compile(python_source, f"{notebook_path}:cell-{index}", "exec")



def test_notebook_setup_adds_repository_root_and_src_to_import_path(monkeypatch):
    notebook_path = Path(__file__).parents[1] / "notebooks" / "02_continual_learning.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "\n".join(notebook["cells"][3]["source"])

    assert "Path.cwd().resolve()" in setup
    assert '(candidate / "pyproject.toml").is_file()' in setup
    assert '(candidate / "src" / "polynomial_cl").is_dir()' in setup
    assert "sys.path.insert(0, str(_import_path))" in setup

    # Exercise the bootstrap from the same working directory used by Jupyter.
    monkeypatch.chdir(notebook_path.parent)
    python_setup = "\n".join(
        line for line in setup.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )
    namespace = {}
    exec(compile(python_setup, str(notebook_path) + ":setup", "exec"), namespace)
    from polynomial_cl.multiclass_data import make_synthetic_2d

    assert callable(make_synthetic_2d)



def test_notebook_uses_package_for_multiclass_learning():
    notebook_path = Path(__file__).parents[1] / "notebooks" / "02_continual_learning.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )

    assert "from polynomial_cl.multiclass_data import make_synthetic_2d" in code
    assert "from polynomial_cl.multiclass_features import polynomial_features_projected" in code
    assert "from experiments.continual_comparison import run_experiment" in code
    assert "synthetic_result = run_experiment(experiment_args(\"synthetic_2d\"))" in code
    assert "split_mnist_result = run_experiment(experiment_args(\"split_mnist\"))" in code
    assert "masking_diagnostics_by_experience" in code
    assert "split_mnist_comparison_v2.json" not in code
    assert "synthetic_2d_comparison.json" not in code
    assert "RESULTS_DIR" not in code
