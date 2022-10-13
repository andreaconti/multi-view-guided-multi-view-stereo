"""
Tests concerning data loading
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.append(".")
from lib.datasets import (
    MVSDataset,
    blended_mvg_utils,
    blended_mvs_utils,
    dtu_utils,
    eth3d_utils,
)
from lib.datasets.sample_preprocess import MVSSampleTransform


def _test_scans(dataset, split):
    if dataset == "dtu":
        ds_utils = dtu_utils
    elif dataset == "blended_mvs":
        ds_utils = blended_mvs_utils
    elif dataset == "blended_mvg":
        ds_utils = blended_mvg_utils
    elif dataset == "eth3d":
        ds_utils = eth3d_utils

    scans = {
        "train": ds_utils.train_scans(),
        "val": ds_utils.val_scans(),
        "test": ds_utils.test_scans(),
    }[split]

    if not scans:
        raise ValueError(f"{split} not implemented for {dataset}")

    path = Path("data/blended-mvs")
    if dataset == "dtu":
        path = Path("data/dtu")
    if dataset == "eth3d":
        path = Path("data/eth3d")

    missing_scans = []
    for scan in scans:
        paths = ds_utils.datapath_files(path, scan, 10, split, 3)

        exist = True
        error = []
        for k, v in paths.items():
            if v is not None and not v.parent.exists():
                exist = False
                error.append(f"not found: {str(v.parent)}")
        if split == "test" and dataset == "dtu":
            assert paths["pcd"] is not None
            assert paths["obs_mask"] is not None
            assert paths["ground_plane"] is not None

        if not exist:
            missing_scans.append((scan, error))

    error_out = ""
    if missing_scans:
        for scan, errors in missing_scans:
            error_out += f"\nerror in scan {scan}: \n" + "".join(f"- {err}\n" for err in errors)
        assert False, error_out


@pytest.mark.dtu
@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dtu_scans(split):
    _test_scans("dtu", split)


@pytest.mark.eth3d
@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_eth3d_scans(split):
    if split in ["train", "val"]:
        with pytest.raises(ValueError):
            _test_scans("eth3d", split)


@pytest.mark.blended_mvs
@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_blended_mvs_scans(split):
    _test_scans("blended_mvs", split)


@pytest.mark.blended_mvg
@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_blended_mvg_scans(split):
    _test_scans("blended_mvg", split)


def _test_ds_loading(name, mode, nviews, ndepths):
    if name == "dtu":
        name = "dtu_yao"

    if name in ["blended_mvs", "blended_mvg"]:
        datapath = "data/blended-mvs"
    elif name == "eth3d":
        datapath = "data/eth3d"
    else:
        datapath = "data/dtu"

    dataset = MVSDataset(
        name,
        datapath=datapath,
        mode=mode,
        nviews=nviews,
        ndepths=ndepths,
    )

    batch = dataset[0]

    for key in {
        "imgs",
        "intrinsics",
        "extrinsics",
        "depths",
        "ref_depth_min",
        "ref_depth_max",
        "ref_depth_values",
        "filename",
    }:
        assert key in batch.keys()
    assert isinstance(batch["imgs"], list)
    assert isinstance(batch["intrinsics"], list)
    assert batch["intrinsics"][0].shape == (3, 3)
    assert isinstance(batch["extrinsics"], list)
    assert batch["extrinsics"][0].shape == (4, 4)

    if name == "dtu_yao":
        img_shape = (512, 640)
        if mode == "test":
            img_shape = (1200, 1600)
        assert batch["depths"][0].shape == (*img_shape, 1)
    elif name in ["blended_mvs", "blended_mvg"]:
        assert batch["depths"][0].shape == (576, 768, 1)

    assert isinstance(batch["depths"], list)
    assert isinstance(batch["ref_depth_min"], float)
    assert isinstance(batch["ref_depth_max"], float)
    assert batch["ref_depth_values"].shape == (ndepths,)
    assert isinstance(batch["filename"], str)


@pytest.mark.eth3d
@pytest.mark.data
@pytest.mark.parametrize(
    "mode, nviews, ndepths",
    itertools.product(
        ["train", "val", "test"],
        [3, 5],
        [192, 128],
    ),
)
def test_eth3d_loading(mode, nviews, ndepths):
    if mode in ["train", "val"]:
        with pytest.raises(ValueError):
            _test_ds_loading("eth3d", mode, nviews, ndepths)
    else:
        _test_ds_loading("eth3d", mode, nviews, ndepths)


@pytest.mark.dtu
@pytest.mark.data
@pytest.mark.parametrize(
    "mode, nviews, ndepths",
    itertools.product(
        ["train", "val", "test"],
        [3, 5],
        [192, 128],
    ),
)
def test_dtu_loading(mode, nviews, ndepths):
    _test_ds_loading("dtu", mode, nviews, ndepths)


@pytest.mark.blended_mvs
@pytest.mark.data
@pytest.mark.parametrize(
    "mode, nviews, ndepths",
    itertools.product(
        ["train", "val", "test"],
        [3, 5],
        [192, 128],
    ),
)
def test_blended_mvs_loading(mode, nviews, ndepths):
    _test_ds_loading("blended_mvs", mode, nviews, ndepths)


@pytest.mark.blended_mvg
@pytest.mark.data
@pytest.mark.parametrize(
    "mode, nviews, ndepths",
    itertools.product(
        ["train", "val", "test"],
        [3, 5],
        [192, 128],
    ),
)
def test_blended_mvg_loading(mode, nviews, ndepths):
    _test_ds_loading("blended_mvg", mode, nviews, ndepths)


@pytest.mark.data
@pytest.mark.dtu
@pytest.mark.parametrize("split", ["train", "val", "test"])
def _test_ds_build_list(ds_name, split, path):

    ds_utils = {
        "dtu": dtu_utils,
        "blended_mvs": blended_mvs_utils,
        "blended_mvg": blended_mvg_utils,
    }[ds_name]

    scans = {
        "train": ds_utils.train_scans(),
        "val": ds_utils.val_scans(),
        "test": ds_utils.test_scans(),
    }[split]

    metas = ds_utils.build_list(path, scans, 3)
    for scan, light_idx, ref_view, _ in metas:
        paths = ds_utils.datapath_files(path, scan, ref_view, split, light_idx)
        for val in paths.values():
            if val is not None:
                assert val.exists(), f"{val} not found"


@pytest.mark.data
@pytest.mark.dtu
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dtu_build_list(split):
    _test_ds_build_list("dtu", split, "data/dtu")


@pytest.mark.data
@pytest.mark.blended_mvs
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_blended_mvs_build_list(split):
    _test_ds_build_list("blended_mvs", split, "data/blended-mvs")


@pytest.mark.data
@pytest.mark.blended_mvg
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_blended_mvg_build_list(split):
    _test_ds_build_list("blended_mvg", split, "data/blended-mvs")


@pytest.mark.data
@pytest.mark.parametrize("hints", ["not_guided", "guided", "mvguided", "mvguided_filtered"])
def test_dataset_preprocess(hints):

    # TODO: add tests for pcd fields

    fake_image = np.random.randint(0, 255, size=(512, 640, 3), dtype=np.uint8)
    sample = {
        "imgs": [Image.fromarray(fake_image)] * 3,
        "intrinsics": [np.random.randn(3, 3).astype(np.float32)] * 3,
        "extrinsics": [np.random.randn(4, 4).astype(np.float32)] * 3,
        "depths": [np.random.rand(512, 640, 1).astype(np.float32) * 1000] * 3,
        "ref_depth_min": 0.1,
        "ref_depth_max": 100.0,
        "ref_depth_values": np.arange(0, 1000, 192).astype(np.float32),
        "filename": "scan1/{}/00000000{}",
        "scan_pcd": None,
        "scan_pcd_obs_mask": None,
        "scan_pcd_bounding_box": None,
        "scan_pcd_resolution": None,
        "scan_pcd_ground_plane": None,
    }

    output = MVSSampleTransform(generate_hints=hints)(sample)

    for i in range(4):
        stage = f"stage_{i}"
        dims = 512 // (2 ** i), 640 // (2 ** i)
        assert output["imgs"][stage].shape == torch.Size([3, 3, *dims])
        assert output["proj_matrices"][stage].shape == torch.Size([3, 4, 4])
        assert output["depth"][stage].shape == torch.Size([1, *dims])

    assert "depth_min" in output
    assert "depth_max" in output
    assert "filename" in output
    if hints == "not_guided":
        assert not hasattr(output, "hints"), output.keys()
    else:
        assert output["hints"].shape == torch.Size([1, 512, 640])
