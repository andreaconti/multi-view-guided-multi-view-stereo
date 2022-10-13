import multiprocessing
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from . import blended_mvg_utils, blended_mvs_utils, dtu_utils, eth3d_utils
from .dtu_blended_mvs import MVSDataset, MVSSample

__all__ = ["MVSDataModule", "MVSSample", "find_dataset_def", "find_scans"]


class MVSDataModule(LightningDataModule):
    def __init__(
        self,
        # selection of the dataset
        name: Literal["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"],
        # args for the dataloader
        batch_size: int = 1,
        # args for the dataset
        **kwargs,
    ):
        super().__init__()
        self._ds_builder = find_dataset_def(name, kwargs.pop("datapath", None))
        self._ds_args = kwargs

        # dataloader args
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.mvs_train = self._ds_builder(mode="train", **self._ds_args)
            self.mvs_val = self._ds_builder(mode="val", **self._ds_args)
        if stage in ("test", None):
            self.mvs_test = self._ds_builder(mode="test", **self._ds_args)

    def train_dataloader(self):
        return DataLoader(
            self.mvs_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count() // 2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mvs_val,
            batch_size=1,
            shuffle=False,
            num_workers=multiprocessing.cpu_count() // 2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mvs_test,
            batch_size=1,
            shuffle=False,
            num_workers=multiprocessing.cpu_count() // 2,
        )


def find_dataset_def(
    name: Literal["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"],
    datapath: Union[Path, str, None] = None,
) -> Callable[..., MVSDataset]:
    assert name in ["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"]
    if datapath is None:
        datapath = {
            "dtu_yao": "data/dtu",
            "blended_mvs": "data/blended-mvs",
            "blended_mvg": "data/blended-mvs",
            "eth3d": "data/eth3d",
        }[name]
    datapath = Path(datapath)

    def builder(*args, **kwargs):
        return MVSDataset(name, datapath, *args, **kwargs)

    return builder


_SCANS = {
    "dtu_yao": {
        "train": dtu_utils.train_scans(),
        "val": dtu_utils.val_scans(),
        "test": dtu_utils.test_scans(),
    },
    "blended_mvs": {
        "train": blended_mvs_utils.train_scans(),
        "val": blended_mvs_utils.val_scans(),
        "test": blended_mvs_utils.test_scans(),
    },
    "blended_mvg": {
        "train": blended_mvg_utils.train_scans(),
        "val": blended_mvg_utils.val_scans(),
        "test": blended_mvg_utils.test_scans(),
    },
    "eth3d": {
        "train": eth3d_utils.train_scans(),
        "val": eth3d_utils.val_scans(),
        "test": eth3d_utils.test_scans(),
    },
}


def find_scans(
    dataset_name: Literal["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"],
    split: Literal["train", "val", "test"],
) -> Optional[List[str]]:
    try:
        return _SCANS[dataset_name][split]
    except KeyError:
        if dataset_name not in ["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"]:
            raise ValueError(f"{dataset_name} not in dtu_utils, blended_mvs, blended_mvg")
        elif split not in ["train", "val", "test"]:
            raise ValueError(f"{split} not in train, val, test")
