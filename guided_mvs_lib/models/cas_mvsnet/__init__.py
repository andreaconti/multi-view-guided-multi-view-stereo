import warnings
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .cas_mvsnet import CascadeMVSNet, cas_mvsnet_loss

__all__ = ["cas_mvsnet_loss", "CascadeMVSNet", "NetBuilder", "SimpleIntefaceNet"]


class SimpleInterfaceNet(nn.Module):
    """
    Simple common interface to call the pretrained models
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = CascadeMVSNet(*args, **kwargs)
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

            # validhints
            validhints = None
            if hints is not None:
                validhints = (hints > 0).to(torch.float32)

            # call
            out = self.model(imgs, proj_matrices, depth_values, hints, validhints)
            self.all_outputs = out
            return out["depth"]["stage_0"]


class NetBuilder(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.model = CascadeMVSNet()

        def loss_func(loss_data, depth_gt, mask):
            return cas_mvsnet_loss(
                loss_data["depth"],
                depth_gt,
                mask,
            )

        self.loss = loss_func

    def forward(self, batch: dict):

        hints, validhints = None, None
        if "hints" in batch:
            hints = batch["hints"]
            validhints = (hints > 0).to(torch.float32)

        return self.model(
            batch["imgs"]["stage_0"],
            batch["proj_matrices"],
            batch["depth_values"],
            hints,
            validhints,
        )
