import os
from pathlib import Path
from typing import List, Literal, Tuple, Union

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
    split: Literal["train", "test", "val"],
    light_id: int = None,
) -> DatasetExamplePaths:
    """
    Takes in input the root of the Blended MVS dataset and returns a dictionary containing
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
    root = Path(datapath)
    return {
        "img": root / f"{scan}/blended_images/{view_id:0>8}.jpg",
        "proj_mat": root / f"{scan}/cams/{view_id:0>8}_cam.txt",
        "depth_mask": root / f"{scan}/rendered_depth_maps/{view_id:0>8}.pfm",
        "depth": root / f"{scan}/rendered_depth_maps/{view_id:0>8}.pfm",
        "pcd": None,
        "obs_mask": None,
        "ground_plane": None,
    }


def read_cam_file(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Reads a file containing the Blended MVS camera intrinsics, extrinsics, max depth and
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
    depth_max = float(lines[11].split()[3])
    return intrinsics, extrinsics, depth_min, depth_max


def read_depth_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the Blended MVS depth mask
    """
    mask = np.array(read_pfm(path)[0], dtype=np.float32) > 0
    return mask


def read_depth(path: Union[str, Path]) -> np.ndarray:
    """
    Loads the depth DTU depth map
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


def train_scans() -> List[str]:
    return [
        "5c1f33f1d33e1f2e4aa6dda4",
        "5bfe5ae0fe0ea555e6a969ca",
        "5bff3c5cfe0ea555e6bcbf3a",
        "58eaf1513353456af3a1682a",
        "5bfc9d5aec61ca1dd69132a2",
        "5bf18642c50e6f7f8bdbd492",
        "5bf26cbbd43923194854b270",
        "5bf17c0fd439231948355385",
        "5be3ae47f44e235bdbbc9771",
        "5be3a5fb8cfdd56947f6b67c",
        "5bbb6eb2ea1cfa39f1af7e0c",
        "5ba75d79d76ffa2c86cf2f05",
        # "5bb7a08aea1cfa39f1a947ab",
        "5b864d850d072a699b32f4ae",
        "5b6eff8b67b396324c5b2672",
        "5b6e716d67b396324c2d77cb",
        "5b69cc0cb44b61786eb959bf",
        "5b62647143840965efc0dbde",
        "5b60fa0c764f146feef84df0",
        "5b558a928bbfb62204e77ba2",
        "5b271079e0878c3816dacca4",
        "5b08286b2775267d5b0634ba",
        "5afacb69ab00705d0cefdd5b",
        "5af28cea59bc705737003253",
        # "5af02e904c8216544b4ab5a2",
        "5aa515e613d42d091d29d300",
        "5c34529873a8df509ae57b58",
        "5c34300a73a8df509add216d",
        "5c1af2e2bee9a723c963d019",
        "5c1892f726173c3a09ea9aeb",
        "5c0d13b795da9479e12e2ee9",
        "5c062d84a96e33018ff6f0a6",
        "5bfd0f32ec61ca1dd69dc77b",
        "5bf21799d43923194842c001",
        "5bf3a82cd439231948877aed",
        "5bf03590d4392319481971dc",
        "5beb6e66abd34c35e18e66b9",
        # "5be883a4f98cee15019d5b83",
        "5be47bf9b18881428d8fbc1d",
        "5bcf979a6d5f586b95c258cd",
        "5bce7ac9ca24970bce4934b6",
        "5bb8a49aea1cfa39f1aa7f75",
        "5b78e57afc8fcf6781d0c3ba",
        # "5b21e18c58e2823a67a10dd8",
        "5b22269758e2823a67a3bd03",
        "5b192eb2170cf166458ff886",
        "5ae2e9c5fe405c5076abc6b2",
        "5adc6bd52430a05ecb2ffb85",
        "5ab8b8e029f5351f7f2ccf59",
        "5abc2506b53b042ead637d86",
        "5ab85f1dac4291329b17cb50",
        "5a969eea91dfc339a9a3ad2c",
        "5a8aa0fab18050187cbe060e",
        # "5a7d3db14989e929563eb153",
        "5a69c47d0d5d0a7f3b2e9752",
        "5a618c72784780334bc1972d",
        "5a6464143d809f1d8208c43c",
        "5a588a8193ac3d233f77fbca",
        "5a57542f333d180827dfc132",
        "5a572fd9fc597b0478a81d14",
        "5a563183425d0f5186314855",
        "5a4a38dad38c8a075495b5d2",
        "5a48d4b2c7dab83a7d7b9851",
        "5a489fb1c7dab83a7d7b1070",
        # "5a48ba95c7dab83a7d7b44ed",
        "5a3ca9cb270f0e3f14d0eddb",
        "5a3cb4e4270f0e3f14d12f43",
        "5a3f4aba5889373fbbc5d3b5",
        "5a0271884e62597cdee0d0eb",
        "59e864b2a9e91f2c5529325f",
        "599aa591d5b41f366fed0d58",
        "59350ca084b7f26bf5ce6eb8",
        "59338e76772c3e6384afbb15",
        "5c20ca3a0843bc542d94e3e2",
        "5c1dbf200843bc542d8ef8c4",
        "5c1b1500bee9a723c96c3e78",
        "5bea87f4abd34c35e1860ab5",
        "5c2b3ed5e611832e8aed46bf",
        "57f8d9bbe73f6760f10e916a",
        "5bf7d63575c26f32dbf7413b",
        # "5be4ab93870d330ff2dce134",
        "5bd43b4ba6b28b1ee86b92dd",
        "5bccd6beca24970bce448134",
        "5bc5f0e896b66a2cd8f9bd36",
        "5b908d3dc6ab78485f3d24a9",
        "5b2c67b5e0878c381608b8d8",
        "5b4933abf2b5f44e95de482a",
        "5b3b353d8d46a939f93524b9",
        "5acf8ca0f3d8a750097e4b15",
        "5ab8713ba3799a1d138bd69a",
        "5aa235f64a17b335eeaf9609",
        # "5aa0f9d7a9efce63548c69a1",
        "5a8315f624b8e938486e0bd8",
        "5a48c4e9c7dab83a7d7b5cc7",
        "59ecfd02e225f6492d20fcc9",
        "59f87d0bfa6280566fb38c9a",
        "59f363a8b45be22330016cad",
        "59f70ab1e5c5d366af29bf3e",
        "59e75a2ca9e91f2c5526005d",
        "5947719bf1b45630bd096665",
        "5947b62af1b45630bd0c2a02",
        "59056e6760bb961de55f3501",
        "58f7f7299f5b5647873cb110",
        "58cf4771d0f5fb221defe6da",
        "58d36897f387231e6c929903",
        "58c4bb4f4a69c55606122be4",
    ]


def val_scans() -> List[str]:
    return [
        "5bb7a08aea1cfa39f1a947ab",
        "5af02e904c8216544b4ab5a2",
        "5be883a4f98cee15019d5b83",
        "5b21e18c58e2823a67a10dd8",
        "5a7d3db14989e929563eb153",
        "5a48ba95c7dab83a7d7b44ed",
        "5be4ab93870d330ff2dce134",
        "5aa0f9d7a9efce63548c69a1",
    ]


def test_scans() -> List[str]:
    return [
        "5b7a3890fc8fcf6781e2593a",
        "5c189f2326173c3a09ed7ef3",
        "5b950c71608de421b1e7318f",
        "5a6400933d809f1d8200af15",
        "59d2657f82ca7774b1ec081d",
        "5ba19a8a360c7c30c1c169df",
        "59817e4a1bd4b175e7038d19",
    ]
