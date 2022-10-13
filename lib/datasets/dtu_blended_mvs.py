import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, TypedDict

import cv2
import numpy as np
import scipy.io
import torch
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset

import lib.datasets.blended_mvg_utils as mvg_utils
import lib.datasets.blended_mvs_utils as mvs_utils
import lib.datasets.dtu_utils as dtu_utils
import lib.datasets.eth3d_utils as eth3d_utils


class MVSSample(TypedDict):
    imgs: List[Image.Image]
    intrinsics: List[np.ndarray]
    extrinsics: List[np.ndarray]
    depths: List[np.ndarray]
    ref_depth_min: float
    ref_depth_max: float
    ref_depth_values: np.ndarray
    filename: str
    scan_pcd: Optional[np.ndarray]
    scan_pcd_obs_mask: Optional[np.ndarray]
    scan_pcd_bounding_box: Optional[np.ndarray]
    scan_pcd_resolution: Optional[float]
    scan_pcd_ground_plane: Optional[np.ndarray]


def _identity_fn(x: MVSSample) -> Dict:
    return x


class MVSDataset(Dataset):
    def __init__(
        self,
        name: Literal["dtu_yao", "blended_mvs", "blended_mvg", "eth3d"],
        datapath: str,
        mode: Literal["train", "val", "test"],
        nviews: int = 5,
        ndepths: int = 192,
        robust_train: bool = False,
        transform: Callable[[Dict], Dict] = _identity_fn,
    ):
        super().__init__()
        assert mode in ["train", "val", "test"], "MVSDataset train, val or test"

        if name == "dtu_yao":
            self.ds_utils = dtu_utils
        elif name == "blended_mvs":
            self.ds_utils = mvs_utils
        elif name == "blended_mvg":
            self.ds_utils = mvg_utils
        elif name == "eth3d":
            self.ds_utils = eth3d_utils
        else:
            raise ValueError("datasets supported: dtu_yao, blended_mvs, blended_mvg")

        self.datapath = datapath
        self.transform = transform
        self.name = name
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.robust_train = robust_train

        scans = {
            "train": self.ds_utils.train_scans,
            "val": self.ds_utils.val_scans,
            "test": self.ds_utils.test_scans,
        }[mode]()
        if scans is None:
            raise ValueError(f"{mode} not supported on dataset {self.name}")

        self.metas = self.ds_utils.build_list(self.datapath, scans, nviews)

    def __len__(self):
        return len(self.metas)

    def _resize_depth_dtu(self, depth, size):
        h, w, _ = depth.shape
        depth = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)[..., None]
        h, w, _ = depth.shape
        start_h, start_w = (h - size[0]) // 2, (w - size[1]) // 2
        depth = depth[start_h : start_h + size[0], start_w : start_w + size[1]]
        return depth

    def __getitem__(self, idx: int) -> MVSSample:

        # load info
        scan, light_idx, ref_view, src_views = self.metas[idx]
        if self.mode in ["train", "val"] and self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
        else:
            view_ids = [ref_view] + src_views[: self.nviews - 1]

        # collect the reference depth, the images and meta for this example
        out = defaultdict(
            lambda: [],
            scan_pcd=None,
            scan_pcd_obs_mask=None,
            scan_pcd_bounding_box=None,
            scan_pcd_resolution=None,
            scan_pcd_ground_plane=None,
        )
        for i, vid in enumerate(view_ids):
            datapaths = self.ds_utils.datapath_files(
                self.datapath, scan, vid, self.mode, light_idx
            )
            out["imgs"].append(Image.open(datapaths["img"]))

            if datapaths["pcd"] is not None and i == 0:
                mesh = PlyData.read(datapaths["pcd"])
                out["scan_pcd"] = np.stack(
                    [mesh["vertex"]["x"], mesh["vertex"]["y"], mesh["vertex"]["z"]], -1
                )
                obs_mask_data = scipy.io.loadmat(datapaths["obs_mask"])
                out["scan_pcd_obs_mask"] = obs_mask_data["ObsMask"]
                out["scan_pcd_bounding_box"] = obs_mask_data["BB"]
                out["scan_pcd_resolution"] = obs_mask_data["Res"]
                out["scan_pcd_ground_plane"] = scipy.io.loadmat(datapaths["ground_plane"])["P"]

            # build proj matrix
            intrinsics, extrinsics, depth_min, depth_max = self.ds_utils.read_cam_file(
                datapaths["proj_mat"]
            )
            if self.name == "dtu_yao" and self.mode != "test":
                # preprocessed images loaded by dtu_yao train and val have been halved in dimension
                # and then cropped to (512, 640)
                intrinsics[:2] *= 0.5
                intrinsics[0, 2] *= 640 / 800
                intrinsics[1, 2] *= 512 / 600
            out["intrinsics"].append(intrinsics)
            out["extrinsics"].append(extrinsics)

            mask = self.ds_utils.read_depth_mask(datapaths["depth_mask"])
            depth = self.ds_utils.read_depth(datapaths["depth"]) * mask

            if self.name == "dtu_yao" and self.mode != "test":
                # the original depth map of dtu must be brought to (512, 640) when in training phase
                mask = self._resize_depth_dtu(mask.astype(np.float32), (512, 640))
                depth = self._resize_depth_dtu(depth, (512, 640))

            out["depths"].append(depth)
            if i == 0:
                out["ref_depth_min"] = depth_min
                out["ref_depth_max"] = depth_max
                out["ref_depth_values"] = np.arange(
                    depth_min,
                    depth_max,
                    (depth_max - depth_min) / self.ndepths,
                    dtype=np.float32,
                )

        out["filename"] = os.path.join(scan, "{}", f"{view_ids[0]:0>8}" + "{}")
        return self.transform(dict(out))
