from collections import defaultdict
from typing import Dict, Literal, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from lib.datasets.dtu_blended_mvs import MVSSample

__all__ = ["MVSSampleTransform"]


class MVSSampleTransform:
    """
    Preprocess each sample computing each view at 4 different scales and converts the
    sample in torch Tensor and compute hints in various ways
    """

    def __init__(
        self,
        generate_hints: Literal[
            "not_guided", "guided", "mvguided", "mvguided_filtered"
        ] = "not_guided",
        hints_perc: float = 0.01,
        filtering_window: Tuple[int, int] = (9, 9),
    ):
        assert generate_hints in ["not_guided", "guided", "mvguided", "mvguided_filtered"]

        self.generate_hints = generate_hints
        self._hints_perc = hints_perc
        self._height_bin = (filtering_window[0] // 2) - 1
        self._width_bin = (filtering_window[1] // 2) - 1

    def _generate_hints(self, sample: Dict) -> torch.Tensor:

        if self.generate_hints == "guided":
            # use only the ref depth map
            depth = sample["depths"][0]
            valid_hints = (depth > 0) & (np.random.rand(*depth.shape) <= self._hints_perc)
            hints = depth * valid_hints
            return torch.from_numpy(hints).permute(2, 0, 1)
        elif self.generate_hints in ["mvguided", "mvguided_filtered"]:

            hints_perc = self._hints_perc

            # extract hints from each depth map
            unwarped_hints = []
            for depth in sample["depths"]:
                valid_hints = (depth > 0) & (np.random.rand(*depth.shape) <= hints_perc)
                unwarped_hints.append(torch.from_numpy(depth * valid_hints).permute(2, 0, 1)[None])

            # project hints from other depth maps in the ref one
            ref_hints, ref_mask = unwarped_hints[0], unwarped_hints[0] > 0
            for i in range(1, len(unwarped_hints)):
                proj_mat_ref = sample["extrinsics"][0].copy()
                proj_mat_ref[:3, :4] = sample["intrinsics"][0] @ proj_mat_ref[:3, :4]
                proj_mat_src = sample["extrinsics"][i].copy()
                proj_mat_src[:3, :4] = sample["intrinsics"][i] @ proj_mat_src[:3, :4]

                warped_hints, warped_mask = hints_homo_warping(
                    unwarped_hints[i],
                    (unwarped_hints[i] > 0).to(torch.float32),
                    torch.from_numpy(proj_mat_ref[None]),
                    torch.from_numpy(proj_mat_src[None]),
                )
                assign_mask = (ref_mask * (1 - warped_mask)) == 0
                ref_hints[assign_mask] = warped_hints[assign_mask]
                ref_mask = ref_mask | warped_mask.to(torch.bool)

            # filter if required
            ref_hints = ref_hints[0] * ref_mask[0]
            if self.generate_hints == "mvguided_filtered":
                hints_mask = outlier_removal_mask(
                    ref_hints.numpy().transpose([1, 2, 0]),
                    sample["intrinsics"][0],
                    height_bin=self._height_bin,
                    width_bin=self._width_bin,
                )
                ref_hints[~torch.from_numpy(hints_mask).to(torch.bool).permute(2, 0, 1)] = 0

            return ref_hints

    def _split_pad(self, pad):
        if pad % 2 == 0:
            return pad // 2, pad // 2
        else:
            pad_1 = pad // 2
            pad_2 = (pad // 2) + 1
            return pad_1, pad_2

    def _pad_to_div_by(self, x, *, div_by=8):

        # compute padding
        if isinstance(x, Image.Image):
            w, h = x.size
        elif isinstance(x, np.ndarray):
            h, w, _ = x.shape
        else:
            raise ValueError("Image or np.ndarray")

        new_h = int(np.ceil(h / div_by)) * div_by
        new_w = int(np.ceil(w / div_by)) * div_by
        pad_t, pad_b = self._split_pad(new_h - h)
        pad_l, pad_r = self._split_pad(new_w - w)

        # return PIL or np.ndarray
        if isinstance(x, Image.Image):
            return F.pad(x, (pad_l, pad_t, pad_r, pad_b))
        elif isinstance(x, np.ndarray):
            return np.pad(x, [(pad_t, pad_b), (pad_l, pad_r), (0, 0)])

    def __call__(self, sample: MVSSample) -> Dict:

        # padding
        imgs = []
        for img, intrins in zip(sample["imgs"], sample["intrinsics"]):

            # pad the image
            w, h = img.size
            imgs.append(self._pad_to_div_by(img, div_by=32))
            w_new, h_new = imgs[-1].size

            # adapt intrinsics
            pad_w = (w_new - w) / 2
            ratio = (w + pad_w) / w
            intrins[0, 2] = intrins[0, 2] * ratio
            pad_h = (h_new - h) / 2
            ratio = (h + pad_h) / h
            intrins[1, 2] = intrins[1, 2] * ratio

        sample["imgs"] = imgs
        sample["depths"] = [self._pad_to_div_by(x, div_by=32) for x in sample["depths"]]

        # compute downsampled stages
        imgs, proj_matrices = defaultdict(lambda: []), defaultdict(lambda: [])
        depths = {}

        w, h = sample["imgs"][0].size
        for i in range(4):
            dsize = (w // (2 ** i), h // (2 ** i))

            for img in sample["imgs"]:
                imgs[f"stage_{i}"].append(F.to_tensor(cv2.resize(np.array(img), dsize)))

            for extrinsics, intrinsics in zip(sample["extrinsics"], sample["intrinsics"]):
                proj_mat = extrinsics.copy()
                intrinsics_copy = intrinsics.copy()
                intrinsics_copy[:2, :] = intrinsics_copy[:2, :] / (2 ** i)
                proj_mat[:3, :4] = intrinsics_copy @ proj_mat[:3, :4]
                proj_matrices[f"stage_{i}"].append(torch.from_numpy(proj_mat))

            depths[f"stage_{i}"] = F.to_tensor(cv2.resize(sample["depths"][0], dsize))

        # return result
        out = {
            "imgs": {k: torch.stack(v) for k, v in imgs.items()},
            "depth": depths,
            "proj_matrices": {k: torch.stack(v) for k, v in proj_matrices.items()},
            "intrinsics": torch.from_numpy(np.stack(sample["intrinsics"])),
            "extrinsics": torch.from_numpy(np.stack(sample["extrinsics"])),
            "depth_min": sample["ref_depth_min"],
            "depth_max": sample["ref_depth_max"],
            "depth_values": torch.from_numpy(sample["ref_depth_values"]),
            "filename": sample["filename"],
        }

        # add hints if requested
        if self.generate_hints != "not_guided":
            out["hints"] = self._generate_hints(sample)

        # add fields pcd if available
        for key in [
            "scan_pcd",
            "scan_pcd_obs_mask",
            "scan_pcd_bounding_box",
            "scan_pcd_resolution",
            "scan_pcd_ground_plane",
        ]:
            if key in sample and sample[key] is not None:
                out[key] = torch.from_numpy(sample[key])

        return out


def _get_all_points(lidar: np.ndarray, intrinsic: np.ndarray) -> Tuple:

    lidar_32 = np.squeeze(lidar).astype(np.float32)
    height, width = np.shape(lidar_32)
    x_axis = np.arange(width).reshape(width, 1)
    x_image = np.tile(x_axis, height)
    x_image = np.transpose(x_image)
    y_axis = np.arange(height).reshape(height, 1)
    y_image = np.tile(y_axis, width)
    z_image = np.ones((height, width))
    image_coor_tensor = (
        np.asarray([x_image, y_image, z_image]).astype(np.float32).transpose([1, 0, 2])
    )

    intrinsic = np.reshape(intrinsic, [3, 3]).astype(np.float32)
    intrinsic_inverse = np.linalg.inv(intrinsic)
    points_homo = np.matmul(intrinsic_inverse, image_coor_tensor)

    lidar_32 = np.reshape(lidar_32, [height, 1, width])
    points_homo = points_homo * lidar_32
    extra_image = np.ones((height, width)).astype(np.float32)
    extra_image = np.reshape(extra_image, [height, 1, width])
    points_homo = np.concatenate([points_homo, extra_image], axis=1)

    extrinsic_v_2_c = [
        [0.007, -1, 0, 0],
        [0.0148, 0, -1, -0.076],
        [1, 0, 0.0148, -0.271],
        [0, 0, 0, 1],
    ]
    extrinsic_v_2_c = np.reshape(extrinsic_v_2_c, [4, 4]).astype(np.float32)
    extrinsic_c_2_v = np.linalg.inv(extrinsic_v_2_c)
    points_lidar = np.matmul(extrinsic_c_2_v, points_homo)

    mask = np.squeeze(lidar) > 0.1
    total_points = [
        points_lidar[:, 0, :][mask],
        points_lidar[:, 1, :][mask],
        points_lidar[:, 2, :][mask],
    ]
    total_points = np.asarray(total_points)
    total_points = np.transpose(total_points)

    return total_points, x_image[mask], y_image[mask], x_image, y_image


def _do_range_projection_try(
    points: np.ndarray,
    fov_up: float = 3.0,
    fov_down: float = -18.0,
) -> Tuple[np.ndarray, np.ndarray]:
    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    return proj_x, proj_y


def _compute_trunck(v: np.ndarray, height_bin: int = 4, width_bin: int = 4) -> np.ndarray:
    v = np.squeeze(v)
    v_trunck = np.lib.stride_tricks.sliding_window_view(
        np.pad(v, [(height_bin, height_bin), (width_bin, width_bin)]),
        (height_bin * 2 + 1, width_bin * 2 + 1),
    ).reshape(v.shape[0], v.shape[1], (height_bin * 2 + 1) * (height_bin * 2 + 1))
    return v_trunck


def _compute_residual(v, height_bin, width_bin):
    v_trunck = _compute_trunck(v, height_bin, width_bin)
    residual = np.squeeze(v)[..., None] - v_trunck
    return residual


def outlier_removal_mask(
    lidar: np.ndarray, intrinsic: np.ndarray, height_bin: int = 4, width_bin: int = 4
) -> np.ndarray:

    height, width, _ = lidar.shape

    total_points, x_indices, y_indices, width_image, height_image = _get_all_points(
        lidar, intrinsic
    )
    proj_x, proj_y = _do_range_projection_try(total_points)

    project_x, project_y = np.zeros((2, height, width, 1))
    project_x[y_indices, x_indices, 0] = proj_x
    project_y[y_indices, x_indices, 0] = proj_y

    project_x_trunck = _compute_trunck(project_x, height_bin, width_bin)
    project_x_residual = project_x - project_x_trunck

    project_y_trunck = _compute_trunck(project_y, height_bin, width_bin)
    project_y_residual = project_y - project_y_trunck

    height_image_trunck = _compute_trunck(height_image, height_bin, width_bin)
    height_image_residual = height_image[..., None] - height_image_trunck

    width_image_trunck = _compute_trunck(width_image, height_bin, width_bin)
    width_image_residual = width_image[..., None] - width_image_trunck

    lidar_trunck = _compute_trunck(lidar, height_bin, width_bin)
    zero_mask = np.logical_and(lidar > 0.1, lidar_trunck > 0.1)

    x_mask = np.logical_and(
        np.logical_or(
            np.logical_and(project_x_residual > 0.0000, width_image_residual <= 0),
            np.logical_and(project_x_residual < 0.0000, width_image_residual >= 0),
        ),
        zero_mask,
    )

    y_mask = np.logical_and(
        np.logical_or(
            np.logical_and(project_y_residual > 0, height_image_residual <= 0),
            np.logical_and(project_y_residual < 0, height_image_residual >= 0),
        ),
        zero_mask,
    )

    lidar_residual = lidar - lidar_trunck
    lidar_mask = np.logical_and(lidar_residual > 3.0, lidar > 0.01)

    final_mask = np.logical_and(lidar_mask, np.logical_or(x_mask, y_mask))
    final_mask = np.squeeze(final_mask)
    final_mask = np.sum(final_mask, axis=-1, keepdims=True) == 0

    return final_mask


def hints_homo_warping(src_hints, src_validhints, src_proj, ref_proj):
    # src_fea: [B, 1, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch = src_hints.shape[0]
    height, width = src_hints.shape[2], src_hints.shape[3]
    warped_hints = torch.zeros_like(src_hints)
    warped_validhints = torch.zeros_like(src_validhints)

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_hints.device),
                torch.arange(0, width, dtype=torch.float32, device=src_hints.device),
            ]
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2) * src_hints.view(
            batch, 1, 1, -1
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]

        proj_x = proj_xyz[:, 0, :, :].view(batch, height, width)
        proj_y = proj_xyz[:, 1, :, :].view(batch, height, width)
        proj_z = proj_xyz[:, 2, :, :].view(batch, height, width)

        proj_x = torch.clamp(np.round(proj_x / proj_z).to(torch.int64), min=0, max=width - 1)
        proj_y = torch.clamp(np.round(proj_y / proj_z).to(torch.int64), min=0, max=height - 1)
        proj_z = proj_z.unsqueeze(-1)

        warped_hints = warped_hints.squeeze(1).unsqueeze(-1)
        warped_validhints = warped_validhints.squeeze(1).unsqueeze(-1)
        src_validhints = src_validhints.squeeze(1).unsqueeze(-1)

        # forward warping (will it work?)
        for i in range(warped_hints.shape[0]):
            warped_hints[i][proj_y, proj_x] = -proj_z[i]
            warped_validhints[i][proj_y, proj_x] = -src_validhints[i]
        warped_hints *= -1
        warped_validhints *= -1 * (warped_hints > 0)

    return warped_hints.permute(0, 3, 1, 2), warped_validhints.permute(
        0, 3, 1, 2
    )  # , warped_validhints
