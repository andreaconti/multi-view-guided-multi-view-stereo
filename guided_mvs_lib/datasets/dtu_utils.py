"""
Utility functions to load the DTU dataset
"""

from pathlib import Path
from typing import Dict, Literal, Tuple, Union

import numpy as np
from PIL import Image

from .utils import DatasetExamplePaths, read_pfm, save_pfm

__all__ = [
    "datapath_files",
    "read_cam_file",
    "read_depth_mask",
    "read_depth",
    "build_list",
    "train_scans",
    "val_scans",
    "test_scans",
]


def datapath_files(
    datapath: Union[Path, str],
    scan: str,
    view_id: int,
    split: Literal["train", "test", "val"],
    light_id: int = None,
) -> DatasetExamplePaths:
    """
    Takes in input the root of the DTU dataset and returns a dictionary containing
    the paths to the files of a single scan, with the speciied view_id and light_id
    (this last one is used only when ``split`` isn't test)

    Parameters
    ----------
    datapath: path
        path to the root of the dtu dataset
    scan: str
        name of the used scan, say scan1
    view_id: int
        the index of the specific image in the scan
    split: train, val or test
        which split of DTU must be used to search the scan for
    light_id: int
        the index of the specific lightning condition index

    Returns
    -------
    out: Dict[str, Path]
        returns a dictionary containing the paths taken into account
    """
    assert split in ["train", "val", "test"]
    root = Path(datapath)
    if split in ["train", "val"]:
        root = root / "train_data"
        return {
            "img": root / f"Rectified/{scan}_train/rect_{view_id + 1:0>3}_{light_id}_r5000.png",
            "proj_mat": root / f"Cameras_1/{view_id:0>8}_cam.txt",
            "depth_mask": root / f"Depths_raw/{scan}/depth_visual_{view_id:0>4}.png",
            "depth": root / f"Depths_raw/{scan}/depth_map_{view_id:0>4}.pfm",
            "pcd": None,
            "obs_mask": None,
            "ground_plane": None,
        }
    else:
        scan_idx = int(scan[4:])
        root = root / "test_data"
        return {
            "img": root / f"{scan}/images/{view_id:0>8}.jpg",
            "proj_mat": root / f"{scan}/cams_1/{view_id:0>8}_cam.txt",
            "depth_mask": root.parent
            / f"train_data/Depths_raw/{scan}/depth_visual_{view_id:0>4}.png",
            "depth": root.parent / f"train_data/Depths_raw/{scan}/depth_map_{view_id:0>4}.pfm",
            "pcd": root.parent / f"SampleSet/MVS Data/Points/stl/stl{scan_idx:0>3}_total.ply",
            "obs_mask": root.parent / f"SampleSet/MVS Data/ObsMask/ObsMask{scan_idx}_10.mat",
            "ground_plane": root.parent / f"SampleSet/MVS Data/ObsMask/Plane{scan_idx}.mat",
        }


def read_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Reads a file containing the DTU camera intrinsics, extrinsics, max depth and
    min depth.

    Parameters
    ----------
    path: str
        path of the source file (something like ../00000000_cam.txt)

    Returns
    -------
    out: Tuple[np.ndarray, np.ndarray, float, float]
        respectively intrinsics, extrinsics, min depth and max depth
    """
    with open(path) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_max


def read_depth_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the depth DTU depth mask
    """
    img = np.array(Image.open(path))[..., None] > 10
    return img


def read_depth(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the depth DTU depth mask
    """
    depth = np.array(read_pfm(str(path))[0], dtype=np.float32)
    return depth


# LIST OF SCANS


def build_list(datapath: Union[str, Path], scans: list, nviews: int):
    metas = []

    datapath = Path(datapath)
    for scan in scans:

        # find pair file
        pair_file_test = datapath / f"test_data/{scan}/pair.txt"
        pair_file_train = datapath / "train_data/Cameras_1/pair.txt"
        if not pair_file_test.exists():
            if not pair_file_train.exists():
                raise ValueError(f"scan {scan} not found")
            else:
                pair_file = pair_file_train
                split = "train"
        else:
            pair_file = pair_file_test
            split = "test"

        # use pair file
        with open(pair_file, "rt") as f:
            num_viewpoint = int(f.readline())
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                if split == "train" and len(src_views) >= nviews:
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
                else:
                    metas.append((scan, None, ref_view, src_views))

    return metas


def train_scans():
    return [
        "scan2",
        "scan6",
        "scan7",
        "scan8",
        "scan14",
        "scan16",
        "scan18",
        "scan19",
        "scan20",
        "scan22",
        "scan30",
        "scan31",
        "scan36",
        "scan39",
        "scan41",
        "scan42",
        "scan44",
        "scan45",
        "scan46",
        "scan47",
        "scan50",
        "scan51",
        "scan52",
        "scan53",
        "scan55",
        "scan57",
        "scan58",
        "scan60",
        "scan61",
        "scan63",
        "scan64",
        "scan65",
        "scan68",
        "scan69",
        "scan70",
        "scan71",
        "scan72",
        "scan74",
        "scan76",
        "scan83",
        "scan84",
        "scan85",
        "scan87",
        "scan88",
        "scan89",
        "scan90",
        "scan91",
        "scan92",
        "scan93",
        "scan94",
        "scan95",
        "scan96",
        "scan97",
        "scan98",
        "scan99",
        "scan100",
        "scan101",
        "scan102",
        "scan103",
        "scan104",
        "scan105",
        "scan107",
        "scan108",
        "scan109",
        "scan111",
        "scan112",
        "scan113",
        "scan115",
        "scan116",
        "scan119",
        "scan120",
        "scan121",
        "scan122",
        "scan123",
        "scan124",
        "scan125",
        "scan126",
        "scan127",
        "scan128",
    ]


def val_scans():
    return [
        "scan3",
        "scan5",
        "scan17",
        "scan21",
        "scan28",
        "scan35",
        "scan37",
        "scan38",
        "scan40",
        "scan43",
        "scan56",
        "scan59",
        "scan66",
        "scan67",
        "scan82",
        "scan86",
        "scan106",
        "scan117",
    ]


def test_scans():
    return [
        "scan1",
        "scan4",
        "scan9",
        "scan10",
        "scan11",
        "scan12",
        "scan13",
        "scan15",
        "scan23",
        "scan24",
        "scan29",
        "scan32",
        "scan33",
        "scan34",
        "scan48",
        "scan49",
        "scan62",
        "scan75",
        "scan77",
        "scan110",
        "scan114",
        "scan118",
    ]
