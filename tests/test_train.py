"""
Testing the training procedure in fast-dev-run with all the networks
on DTU
"""

import shutil
import sys
import tempfile
from pathlib import Path

import py
import pytest

# tests are executed in the root dir working directory, the root directory
# however is not a python package, this hack solves it and allows us to load
# the training script as a module even if this testing file is in `tests` folder
sys.path.append(".")
from train import run_training


@pytest.mark.slow
@pytest.mark.train
@pytest.mark.parametrize(
    "model",
    [
        pytest.param("mvsnet", marks=pytest.mark.mvsnet),
        pytest.param("cas_mvsnet", marks=pytest.mark.cas_mvsnet),
        pytest.param("ucsnet", marks=pytest.mark.ucsnet),
        pytest.param("patchmatchnet", marks=pytest.mark.patchmatchnet),
        pytest.param("d2hc_rmvsnet", marks=pytest.mark.d2hc_rmvsnet),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("dtu_yao", marks=pytest.mark.dtu),
        pytest.param("blended_mvs", marks=[pytest.mark.blended_mvs, pytest.mark.blended_mvg]),
    ],
)
@pytest.mark.parametrize("hints", ["not_guided", "guided"])
def test_network_train(model, dataset, hints):

    # create a folder for the output
    path = tempfile.mkdtemp(prefix="guided-mvs-test-")

    run_training(
        params={
            "model": model,
            "train": {
                "dataset": dataset,
                "epochs": 10,
                "steps": None,
                "batch_size": 1,
                "lr": 0.001,
                "epochs_lr_decay": None,
                "epochs_lr_gamma": 2,
                "weight_decay": 0.0,
                "ndepths": 192,
                "views": 3,
                "hints": hints,
                "hints_density": 0.01,
                "hints_filter_window": [9, 9],
            },
        },
        cmdline_args=["--gpus", "1", "--fast-dev-run"],
        outpath=Path(path),
        logspath=Path(path),
    )

    shutil.rmtree(path, ignore_errors=True)
