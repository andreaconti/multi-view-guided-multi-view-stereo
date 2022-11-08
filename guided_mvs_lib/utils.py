import multiprocessing
from collections import defaultdict
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import cv2
import joblib
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.utils.tensorboard as tb
import torchvision.transforms.functional as F
import torchvision.utils as vutils
from PIL import Image
from scipy.spatial import cKDTree


def prune_pcd(
    pcd: np.ndarray,
    max_distance: float,
    *,
    n_jobs: int = 1,
    verbose: int = 0,
) -> np.ndarray:
    """
    Takes a N x M point cloud and for each point removes the points in its neighbourhood
    nearer than ``distance``.

    Parameters
    ----------
    pcd: array like
        the N x M point cloud composed by N points of M dimensions
    distance: float
        the maximum distance, points nearer thant this value will be collapsed
    n_jobs: int, default 1
        if > 1 the pcd is splitted into ``n_jobs`` chunks, the KD-Tree is computed on each
        of them and at the end a final KD-Tree is computed on the results

    Returns
    -------
    out: boolean array like
        returns a mask of the original point cloud
    """

    def _prune_pcd(pcd, max_distance):
        mask = np.ones((pcd.shape[0],), dtype=np.bool)
        kdtree = cKDTree(pcd)
        idx = kdtree.query_ball_tree(kdtree, max_distance)
        for i, neighbors in enumerate(idx):
            mask[neighbors] = False
            mask[i] = True
        return mask

    if n_jobs > 1:

        # split the pcd
        split_size = len(pcd) // n_jobs
        splits = [split_size * i for i in range(1, n_jobs)]

        # concurrently call kdtrees
        masks = joblib.Parallel(n_jobs, verbose=verbose)(
            joblib.delayed(_prune_pcd)(split, max_distance) for split in np.split(pcd, splits)
        )

        mask = np.concatenate(masks, axis=0)
        sub_mask = _prune_pcd(pcd[mask], max_distance)
        final_mask = np.zeros((pcd.shape[0],), dtype=np.bool)
        final_mask[mask] = sub_mask
        return final_mask
    else:
        return _prune_pcd(pcd, max_distance)


def pcd_distances(
    pts_from: np.ndarray,  # N x 3
    pts_to: np.ndarray,  # N x 3
    bounding_box: np.ndarray,  # 2 x 3
    max_distance: float,
    n_jobs: int = multiprocessing.cpu_count(),
) -> np.ndarray:
    """
    for each voxel in the point cloud delimited by the bounding box it filters the points in ptsFrom
    and searches for points in ptsTo inside the maximum allowed distance, then if there aren't any
    points in ptsTo it assignes the maximum error to each ptsFrom involved (maxdist) otherwise it
    searches for the nearest point of each ptsFrom involved and records the distance

    Parameters
    ----------
    pts_from: array like
        array of shape Nx3 containing the source point cloud
    pts_to: array like
        array of shape Nx3 containing the destination point cloud
    bounding_box: array like
        array of shape 2x3 containing the bounding box for the 3 dimensions x, y, z. For instance
        the x axis bounds are ``(bounding_box[0, 0], bounding_box[1, 0])``
    max_distance: float
        the maximum distance taken into account to search nearest neighbors
    n_jobs: int, defaults to the number of cpu cores
        number of jobs to be used for each voxel and kd-tree search

    Results
    -------
    dist: np.ndarray
        an array of shape N containing for each point in pts_from the distance among it and its nearest
        point in pts_to limited to ``max_distance``
    """

    rx, ry, rz = np.floor((bounding_box[1, :] - bounding_box[0, :]) / max_distance).astype(int)
    dist = np.ones(pts_from.shape[0]) * max_distance

    def distances_in_voxel(x, y, z):

        low = bounding_box[0, :] + np.array([x, y, z]) * max_distance
        high = low + max_distance
        validsFrom = (pts_from >= low[None]).all(axis=1) & (pts_from < high[None]).all(axis=1)

        low = low - max_distance
        high = high + max_distance
        validsTo = (pts_to >= low[None]).all(axis=1) & (pts_to < high[None]).all(axis=1)

        if validsTo.sum() == 0:
            return max_distance, validsFrom
        elif validsFrom.sum() == 0:
            return max_distance, validsFrom
        else:
            kdtree = cKDTree(pts_to[validsTo])
            nearest_dists = kdtree.query(
                pts_from[validsFrom], workers=n_jobs, distance_upper_bound=max_distance
            )[0]
            nearest_dists[nearest_dists == float("inf")] = max_distance
            return nearest_dists, validsFrom

    jobs = []
    for x in range(rx + 1):
        for y in range(ry + 1):
            for z in range(rz + 1):
                jobs.append(joblib.delayed(distances_in_voxel)(x, y, z))
    result = joblib.Parallel(n_jobs=n_jobs)(jobs)
    for nearest_dists, validsFrom in result:
        dist[validsFrom] = nearest_dists

    return dist


def project_depth(
    depth: np.ndarray,
    intrinsics_from: np.ndarray,
    intrinsics_to: np.ndarray,
    extrinsics_from_to: np.ndarray,
    *,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Takes a depth map, project into the 3D space move it over the new point of view
    and projects it again over the new image plane.

    Parameters
    ----------
    depth: array like
        depth map of shape H x W x 1
    intrinsics_from: array like
        intrinsics of shape 3 x 3
    intrinsics_to: array like
        intrinsics of shape 3 x 3
    extrinsics_from_to: array like
        extrinsics R|T of shape 4 x 4
    output_size: tuple
        H and W of the output depth map, by default it is computed using ``intrinsics_to``

    Returns
    -------
    out: array like
        the projected depth map
    """

    # handle optional params
    if output_size is None:
        h, w = int(np.round(intrinsics_to[1, 2])), int(np.round(intrinsics_to[0, 2]))
    else:
        h, w = depth.shape[:2]

    # project the depth in 3D
    v, u, _ = np.nonzero(depth)
    z = depth[v, u, 0]
    xyz_pcd_from = np.linalg.inv(intrinsics_from) @ (np.vstack([u, v, np.ones_like(u)]) * z)

    # project the 3D pcd onto the new image plane
    xyz_pcd_to = (
        intrinsics_to
        @ (extrinsics_from_to @ np.concatenate([xyz_pcd_from, np.ones([1, len(u)])]))[:3]
    )
    u_to, v_to = xyz_pcd_to[:2] / xyz_pcd_to[2]
    u_to, v_to = np.round(u_to).astype(int), np.round(v_to).astype(int)
    mask = (u_to >= 0) & (u_to < w) & (v_to >= 0) & (v_to < h)
    u_to, v_to, z = u_to[mask], v_to[mask], z[mask]
    depth_to = np.zeros([h, w, 1])
    depth_to[v_to, u_to, 0] = z

    return depth_to


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))

    return intrinsics, extrinsics


# read an image
def read_img(filename, img_wh=None):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.0
    if img_wh is not None:
        np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the reference points to the source view, then project back to calculate the reprojection error

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)

    Returns:
        A tuble contains
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x_reprojected: reprojected x coordinates of points in the reference view, of shape (H, W)
            y_reprojected: reprojected y coordinates of points in the reference view, of shape (H, W)
            x_src: x coordinates of points in the source view, of shape (H, W)
            y_src: y coordinates of points in the source view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]),
    )
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
        np.vstack((xyz_ref, np.ones_like(x_ref))),
    )[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]),
    )
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
        np.vstack((xyz_src, np.ones_like(x_ref))),
    )[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
    geo_pixel_thres: float,
    geo_depth_thres: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check geometric consistency and return valid points

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)
        geo_pixel_thres: geometric pixel threshold
        geo_depth_thres: geometric depth threshold

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mask: mask for points with geometric consistency, of shape (H, W)
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x2d_src: x coordinates of points in the source view, of shape (H, W)
            y2d_src: y coordinates of points in the source view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
    )
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def print_args(args: Any) -> None:
    """Utilities to print arguments

    Arsg:
        args: arguments to pring out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def make_nograd_func(func: Callable) -> Callable:
    """Utilities to make function no gradient

    Args:
        func: input function

    Returns:
        no gradient function wrapper for input function
    """

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def make_recursive_func(func: Callable) -> Callable:
    """Convert a function into recursive style to handle nested dict/list/tuple variables

    Args:
        func: input function

    Returns:
        recursive style function
    """

    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars: Any) -> float:
    """Convert tensor to float"""
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars: Any) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars: Any) -> Union[str, torch.Tensor]:
    """Convert tensor to tensor on GPU"""
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


def save_scalars(
    logger: tb.SummaryWriter, mode: str, scalar_dict: Dict[str, Any], global_step: int
) -> None:
    """Log values stored in the scalar dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        scalar_dict: python dictionary stores the key and value pairs to be recorded
        global_step: step index where the logger should write
    """
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def colormap_image(imgs, cmap="magma", valid=None):
    """Apply colormap to image pixels"""
    res = []
    for i in range(imgs.shape[0]):
        x = imgs[i]
        if valid is not None:
            v = valid[i] > 0
        else:
            v = (x * 0) == 0
        ma = float(x[v].max().cpu().data)
        mi = float(x[v].min().cpu().data)
        normalizer = mpl.colors.Normalize(vmin=mi, vmax=ma)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        colormapped_im = (mapper.to_rgba(x.cpu().data[0])[:, :, :3]).astype(np.float32)
        res.append(torch.from_numpy(np.transpose(colormapped_im, (2, 0, 1))))

    return torch.stack(res, 0)


def save_images(
    logger: tb.SummaryWriter, mode: str, images_dict: Dict[str, Any], global_step: int
) -> None:
    """Log images stored in the image dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        images_dict: python dictionary stores the key and image pairs to be recorded
        global_step: step index where the logger should write
    """
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError(
                "invalid img shape {}:{} in save_images".format(name, img.shape)
            )
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter:
    """Wrapper class for dictionary variables that require the average value"""

    def __init__(self) -> None:
        """Initialization method"""
        self.data: Dict[Any, float] = {}
        self.count = 0

    def update(self, new_input: Dict[Any, float]) -> None:
        """Update the stored dictionary with new input data

        Args:
            new_input: new data to update self.data
        """
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self) -> Any:
        """Return the average value of values stored in self.data"""
        return {k: v / self.count for k, v in self.data.items()}


def compute_metrics_for_each_image(metric_func: Callable) -> Callable:
    """A wrapper to compute metrics for each image individually"""

    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper
