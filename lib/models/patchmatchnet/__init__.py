import warnings
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .net import PatchmatchNet, patchmatchnet_loss

__all__ = ["PatchmatchNet", "patchmatchnet_loss", "NetBuilder", "SimpleInterfaceNet"]


class SimpleInterfaceNet(nn.Module):
    """
    Simple common interface to call the pretrained models
    """

    def __init__(self, **kwargs):
        super().__init__()

        default_args = dict(
            patchmatch_interval_scale=[0.005, 0.0125, 0.025],
            propagation_range=[6, 4, 2],
            patchmatch_iteration=[1, 2, 2],
            patchmatch_num_sample=[8, 8, 16],
            propagate_neighbors=[0, 8, 16],
            evaluate_neighbors=[9, 9, 9],
        )
        default_args.update(kwargs)
        self.model = PatchmatchNet(**default_args)
        self.all_outputs: dict[str, Any] = {}

    def forward(
        self,
        imgs: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        depth_values: Tensor,
        hints: Optional[Tensor] = None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # compute poses
            proj_matrices = {}
            for i in range(4):
                proj_mat = extrinsics.clone()
                intrinsics_copy = intrinsics.clone()
                intrinsics_copy[..., :2, :] = intrinsics_copy[..., :2, :] / (2 ** i)
                proj_mat[..., :3, :4] = intrinsics_copy @ proj_mat[..., :3, :4]
                proj_matrices[f"stage_{i}"] = proj_mat

            # downsample images
            imgs_stages = {}
            h, w = imgs.shape[-2:]
            for i in range(4):
                dsize = h // (2 ** i), w // (2 ** i)
                imgs_stages[f"stage_{i}"] = torch.stack(
                    [
                        F.interpolate(img, dsize, mode="bilinear", align_corners=True)
                        for img in torch.unbind(imgs, 1)
                    ],
                    1,
                )

            # validhints
            validhints = None
            if hints is not None:
                validhints = (hints > 0).to(torch.float32)

            # call
            out = self.model(
                imgs_stages,
                proj_matrices,
                depth_values.min(1).values,
                depth_values.max(1).values,
                hints,
                validhints,
            )
            self.all_outputs = out
            return out["depth"]["stage_0"]


class NetBuilder(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()

        self.model = PatchmatchNet(
            patchmatch_interval_scale=[0.005, 0.0125, 0.025],
            propagation_range=[6, 4, 2],
            patchmatch_iteration=[1, 2, 2],
            patchmatch_num_sample=[8, 8, 16],
            propagate_neighbors=[0, 8, 16],
            evaluate_neighbors=[9, 9, 9],
        )

        def loss_func(loss_data, depth_gt, mask):
            return patchmatchnet_loss(
                loss_data["depth_patchmatch"],
                loss_data["refined_depth"],
                depth_gt,
                mask,
            )

        self.loss = loss_func

        self.hparams = {
            "interval_scale": [0.005, 0.0125, 0.025],
            "propagation_range": [6, 4, 2],
            "iteration": [1, 2, 2],
            "num_sample": [8, 8, 16],
            "propagate_neighbors": [0, 8, 16],
            "evaluate_neighbors": [9, 9, 9],
        }

    def forward(self, batch: dict):

        hints, validhints = None, None
        if "hints" in batch:
            hints = batch["hints"]
            validhints = (hints > 0).to(torch.float32)

        return self.model(
            batch["imgs"],
            batch["proj_matrices"],
            batch["depth_min"],
            batch["depth_max"],
            hints,
            validhints,
        )
