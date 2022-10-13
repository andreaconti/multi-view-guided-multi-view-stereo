"""
Tests network interface and operational behaviour
"""

import logging
import sys

sys.path.append(".")
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lib import models
from lib.datasets import MVSDataModule
from lib.datasets.sample_preprocess import MVSSampleTransform

logging.basicConfig(level=logging.DEBUG)

_train_args = {
    "epochs": 5,
    "steps": None,
    "batch_size": 1,
    "lr": 0.001,
    "epochs_lr_decay": None,
    "epochs_lr_gamma": 2,
    "weight_decay": 0.0,
    "ndepths": 128,
    "wiews": 2,
    "hints_density": 0.01,
}


@pytest.mark.slow
@pytest.mark.parametrize("hints", ["not_guided", "guided", "mvguided", "mvguided_filtered"])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("blended_mvg", marks=pytest.mark.blended_mvg),
        pytest.param("blended_mvs", marks=pytest.mark.blended_mvs),
        pytest.param("dtu_yao", marks=pytest.mark.dtu),
        pytest.param("eth3d", marks=pytest.mark.eth3d),
    ],
)
@pytest.mark.parametrize(
    "network",
    [
        pytest.param("mvsnet", marks=pytest.mark.mvsnet),
        pytest.param("cas_mvsnet", marks=pytest.mark.cas_mvsnet),
        pytest.param("ucsnet", marks=pytest.mark.ucsnet),
        pytest.param("patchmatchnet", marks=pytest.mark.patchmatchnet),
        pytest.param("d2hc_rmvsnet", marks=pytest.mark.d2hc_rmvsnet),
    ],
)
def test_network_forward(dataset, network, hints):

    data_module = MVSDataModule(
        dataset,
        nviews=3,
        ndepths=128,
        transform=MVSSampleTransform(generate_hints=hints),
    )
    data_module.setup(stage="test")
    dl = data_module.test_dataloader()
    batch = next(iter(dl))

    # load model
    args = SimpleNamespace(model=network)
    args.train = SimpleNamespace(**_train_args, dataset=dataset, hints=hints)
    network = models.build_network(network, args)

    # mocking batch dictionary for inspection
    batch_mock = MagicMock()
    batch_mock.__getitem__.side_effect = batch.__getitem__
    batch_mock.__contains__.side_effect = batch.__contains__

    # assert output interface and hints usage
    output = network(batch_mock)
    assert set(output.keys()) == {"depth", "photometric_confidence", "loss_data"}
    if "hints" in batch:
        batch_mock.__getitem__.assert_any_call("hints")
