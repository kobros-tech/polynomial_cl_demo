# Copyright (c) 2026 Mohamed Alkobrosli
# SPDX-License-Identifier: MIT

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / name), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
    )


def test_class_incremental_script_writes_a_self_describing_json_result(tmp_path):
    out = tmp_path / "result.json"
    run_script(
        "class_incremental.py", "--dataset", "synthetic_2d", "--seeds", 0, 1, "--epochs", 1,
        "--memory-per-class", 2, "--train-per-class", 6, "--test-per-class", 4, "--output", out,
    )  # fmt: skip
    result = json.loads(out.read_text())
    assert result["dataset"] == "synthetic_2d"
    assert result["protocol"]["seeds"] == [0, 1]
    assert result["protocol"]["degrees_by_condition"]["expanding"] == [1, 2, 3, 3, 3]
    assert len(result["runs"]) == 6
    assert set(result["summary"]) == {"fixed_linear", "expanding", "fixed_capacity"}
    first = result["runs"][0]["accuracy_matrix"]
    assert (
        first[0][1] is None and first[1][1] is not None
    )  # unseen experiences are null


def test_class_incremental_script_fails_cleanly_without_mnist_files(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "class_incremental.py"), "--dataset",
         "split_mnist", "--no-download", "--data-dir", str(tmp_path), "--output",
         str(tmp_path / "x.json")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert proc.returncode != 0 and "download is disabled" in proc.stderr
    assert not (tmp_path / "x.json").exists()


def test_degree_ablation_script_reports_all_three_conditions(tmp_path):
    out = tmp_path / "ablation.json"
    run_script(
        "synthetic_degree_ablation.py", "--seeds", 0, "--n-experiences", 3, "--epochs", 5,
        "--output", out,
    )  # fmt: skip
    result = json.loads(out.read_text())
    assert result["protocol"]["fixed_degree"] == 3
    assert set(result["summary"]) == {
        "expanding",
        "fixed_initial_degree",
        "fixed_capacity",
    }
    assert len(result["runs"]) == 3


def test_class_incremental_script_strategies_mode(tmp_path):
    out = tmp_path / "strategies.json"
    run_script(
        "class_incremental.py", "--dataset", "synthetic_2d", "--experiment", "strategies",
        "--seeds", 0, "--epochs", 1, "--memory-per-class", 2, "--train-per-class", 6,
        "--test-per-class", 4, "--n-ref-per-class", 1, "--output", out,
    )  # fmt: skip
    result = json.loads(out.read_text())
    assert result["protocol"]["experiment"] == "strategies"
    assert result["protocol"]["n_ref_per_class"] == 1
    assert set(result["summary"]) == {
        "ordinary", "ordinary_no_replay", "small_old_lr", "frozen_old", "strategy_b"
    }  # fmt: skip
