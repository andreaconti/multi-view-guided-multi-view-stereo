import tarfile
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal, Tuple

import torch
import yaml

from guided_mvs_lib import __version__ as CURR_VERS
from guided_mvs_lib import models

dependencies = ["torch", "torchvision"]

## entry points for each single model

CURR_DIR = Path(__file__).parent


def _get_archive() -> str:
    if not (CURR_DIR / "trained_models.tar.gz").exists():
        torch.hub.download_url_to_file(
            f"https://github.com/andreaconti/multi-view-guided-multi-view-stereo/releases/download/v{CURR_VERS}/trained_models.tar.gz",
            str(CURR_DIR / "trained_models.tar.gz"),
        )
    return str(CURR_DIR / "trained_models.tar.gz")


def _load_model(
    tarpath: str,
    model: Literal["ucsnet", "d2hc_rmvsnet", "mvsnet", "patchmatchnet", "cas_mvsnet"] = "mvsnet",
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    Utility function to load from the tarfile containing all the pretrained models the one choosen
    """

    assert model in [
        "ucsnet",
        "d2hc_rmvsnet",
        "mvsnet",
        "patchmatchnet",
        "cas_mvsnet",
    ]
    assert dataset in ["blended_mvg", "dtu_yao_blended_mvg"]
    assert hints in ["mvguided_filtered", "not_guided", "guided", "mvguided"]

    # model instance
    model_net = models.__dict__[model].SimpleInterfaceNet()
    model_net.train_params = None

    # find the correct checkpoint
    if pretrained:
        with tarfile.open(tarpath) as archive:
            info = yaml.safe_load(archive.extractfile("trained_models/info.yaml"))
            for ckpt_id, meta in info.items():
                found = meta["model"] == model and meta["hints"] == hints
                if hints != "not_guided":
                    found = found and float(meta["hints_density"]) == hints_density
                if dataset == "blended_mvg":
                    found = found and meta["dataset"] == dataset
                else:
                    found = (
                        found
                        and meta["dataset"] == "dtu_yao"
                        and "load_weights" in meta
                        and info[meta["load_weights"]]["dataset"] == "blended_mvg"
                    )
                if found:
                    break
            if not found:
                raise ValueError("Model not available with the provided parameters")

            model_net.load_state_dict(
                {
                    ".".join(n.split(".")[1:]): v
                    for n, v in torch.load(archive.extractfile(f"trained_models/{ckpt_id}.ckpt"))[
                        "state_dict"
                    ].items()
                }
            )
            model_net.train_params = meta
    return model_net


def mvsnet(
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    pretrained `MVSNet`_ network.

    .. _MVSNet https://arxiv.org/pdf/1804.02505.pdf
    """
    return _load_model(
        _get_archive(),
        "mvsnet",
        pretrained=pretrained,
        dataset=dataset,
        hints=hints,
        hints_density=hints_density,
    )


def ucsnet(
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    pretrained `UCSNet`_ network.

    .. _UCSNet https://arxiv.org/pdf/1911.12012.pdf
    """
    return _load_model(
        _get_archive(),
        "ucsnet",
        pretrained=pretrained,
        dataset=dataset,
        hints=hints,
        hints_density=hints_density,
    )


def d2hc_rmvsnet(
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    pretrained `D2HCRMVSNet`_ network.

    .. _D2HCRMVSNet https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490647.pdf
    """
    return _load_model(
        _get_archive(),
        "d2hc_rmvsnet",
        pretrained=pretrained,
        dataset=dataset,
        hints=hints,
        hints_density=hints_density,
    )


def patchmatchnet(
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    pretrained `PatchMatchNet`_ network.

    .. _PatchMatchNet https://github.com/FangjinhuaWang/PatchmatchNet
    """
    return _load_model(
        _get_archive(),
        "patchmatchnet",
        pretrained=pretrained,
        dataset=dataset,
        hints=hints,
        hints_density=hints_density,
    )


def cas_mvsnet(
    pretrained: bool = True,
    dataset: Literal["blended_mvg", "dtu_yao_blended_mvg"] = "blended_mvg",
    hints: Literal["mvguided_filtered", "not_guided", "guided", "mvguided"] = "not_guided",
    hints_density: float = 0.03,
):
    """
    pretrained `CASMVSNet`_ network.

    .. _CASMVSNet https://arxiv.org/pdf/1912.06378.pdf
    """
    return _load_model(
        _get_archive(),
        "cas_mvsnet",
        pretrained=pretrained,
        dataset=dataset,
        hints=hints,
        hints_density=hints_density,
    )


## Datasets


def _load_dataset(
    dataset: str,
    root: str,
    batch_size: int = 1,
    nviews: int = 5,
    ndepths: int = 128,
    hints: str = "mvguided_filtered",
    hints_density: float = 0.03,
    filtering_window: Tuple[int, int] = (9, 9),
    num_workers: int = cpu_count() // 2,
):
    from guided_mvs_lib.datasets import MVSDataModule
    from guided_mvs_lib.datasets.sample_preprocess import MVSSampleTransform

    dm = MVSDataModule(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        datapath=root,
        nviews=nviews,
        ndepths=ndepths,
        robust_train=False,
        transform=MVSSampleTransform(
            generate_hints=hints,
            hints_perc=hints_density,
            filtering_window=filtering_window,
        ),
    )
    return dm


def blended_mvs(
    root: str,
    batch_size: int = 1,
    nviews: int = 5,
    ndepths: int = 128,
    hints: str = "mvguided_filtered",
    hints_density: float = 0.03,
    filtering_window: Tuple[int, int] = (9, 9),
    num_workers: int = cpu_count() // 2,
):
    """
    Utility function to load a Pytorch Lightning DataModule loading
    the BlendedMVS dataset
    """
    return _load_dataset(
        "blended_mvs",
        root=root,
        batch_size=batch_size,
        nviews=nviews,
        ndepths=ndepths,
        hints=hints,
        hints_density=hints_density,
        filtering_window=filtering_window,
        num_workers=num_workers,
    )


def blended_mvg(
    root: str,
    batch_size: int = 1,
    nviews: int = 5,
    ndepths: int = 128,
    hints: str = "mvguided_filtered",
    hints_density: float = 0.03,
    filtering_window: Tuple[int, int] = (9, 9),
    num_workers: int = cpu_count() // 2,
):
    """
    Utility function to load a Pytorch Lightning DataModule loading
    the BlendedMVG dataset
    """
    return _load_dataset(
        "blended_mvg",
        root=root,
        batch_size=batch_size,
        nviews=nviews,
        ndepths=ndepths,
        hints=hints,
        hints_density=hints_density,
        filtering_window=filtering_window,
        num_workers=num_workers,
    )


def dtu(
    root: str,
    batch_size: int = 1,
    nviews: int = 5,
    ndepths: int = 128,
    hints: str = "mvguided_filtered",
    hints_density: float = 0.03,
    filtering_window: Tuple[int, int] = (9, 9),
    num_workers: int = 4,  # (pretty memory aggressive)
):
    """
    Utility function to load a Pytorch Lightning DataModule loading
    the DTU dataset
    """
    return _load_dataset(
        "dtu_yao",
        root=root,
        batch_size=batch_size,
        nviews=nviews,
        ndepths=ndepths,
        hints=hints,
        hints_density=hints_density,
        filtering_window=filtering_window,
        num_workers=num_workers,
    )
