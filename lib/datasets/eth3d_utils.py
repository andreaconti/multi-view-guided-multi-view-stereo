import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from .utils import DatasetExamplePaths, read_pfm

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
    split: Literal["train", "test", "val"] = "test",
    light_id: int = None,
) -> DatasetExamplePaths:
    """
    Takes in input the root of the ETH3D training dataset and returns a dictionary containing
    the paths to the files of a single scan.

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


    .. note::
        this function is implemented only for the test split and light_id is ignored.
    """

    if split != "test":
        raise ValueError("only test split is implemented for ETH3D")

    root = Path(datapath)
    mapping = open(f"{root}/{scan}/cams/index2prefix.txt").readlines()
    img_name = mapping[view_id].rstrip().split(" ")[1].split(".")[0]
    depth_name = img_name.replace("_undistorted", "")
    return {
        "img": root / f"{scan}/images/{img_name}.png",
        "proj_mat": root / f"{scan}/cams/{view_id:0>8}_cam.txt",
        "depth_mask": root / f"{scan}/depths/{depth_name}.pfm",
        "depth": root / f"{scan}/depths/{depth_name}.pfm",
        "pcd": None,
        "obs_mask": None,
        "ground_plane": None,
    }


def read_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Reads a file containing the ETH3D camera intrinsics, extrinsics, max depth and
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
    interval = float(lines[11].split()[1])
    depth_num = float(lines[11].split()[2])
    depth_max = depth_min + (interval * depth_num)
    return intrinsics, extrinsics, depth_min, depth_max


def read_depth_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the ETH3D depth mask
    """
    mask = np.array(read_pfm(path)[0], dtype=np.float32) > 0
    return mask


def read_depth(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the ETH3D depth map
    """
    depth = np.array(read_pfm(str(path))[0], dtype=np.float32)
    return depth


def build_list(datapath: str, scans: list, nviews: int):
    metas = []
    for scan in scans:
        pair_file = "cams/pair.txt"

        with open(os.path.join(datapath, scan, pair_file)) as f:
            num_viewpoint = int(f.readline())
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                if len(src_views) >= nviews:
                    metas.append((scan, None, ref_view, src_views))
    return metas


def train_scans() -> Optional[List[str]]:
    return None


def val_scans() -> Optional[List[str]]:
    return None


def test_scans() -> Optional[List[str]]:
    return [
        "delivery_area",
        "electro",
        "forest",
        "playground",
        "terrains",
    ]
