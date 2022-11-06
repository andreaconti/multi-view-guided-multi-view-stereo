from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .mvsnet import MVSNet, mvsnet_loss

__all__ = ["mvsnet_loss", "MVSNet", "NetBuilder", "SimpleInterfaceNet"]


class SimpleInterfaceNet(nn.Module):
    """
    Simple common interface to call the pretrained models
    """

    def __init__(self, refine: bool = False):
        super().__init__()
        self.model = MVSNet(refine=refine)
        self.all_outputs: dict[str, Any] = {}

    def forward(
        self,
        imgs: Tensor,
        intrinsics: Tensor,
        extrinsics: Tensor,
        depth_values: Tensor,
        hints: Optional[Tensor] = None,
    ):
        # compute poses
        proj_mat = extrinsics.clone()
        intrinsics_copy = intrinsics.clone()
        intrinsics_copy[..., :2, :] = intrinsics_copy[..., :2, :] / 4
        proj_mat[..., :3, :4] = intrinsics_copy @ proj_mat[..., :3, :4]

        # validhints
        validhints = None
        if hints is not None:
            validhints = (hints > 0).to(torch.float32)

        # call
        out = self.model(imgs, proj_mat, depth_values, hints, validhints)
        self.all_outputs = out
        return out["depth"]["stage_0"]


class NetBuilder(nn.Module):
    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.model = MVSNet(refine=False)
        self.loss = mvsnet_loss

    def forward(self, batch: dict):

        hints, validhints = None, None
        if "hints" in batch:
            hints = batch["hints"]
            validhints = (hints > 0).to(torch.float32)

        return self.model(
            batch["imgs"]["stage_0"],
            batch["proj_matrices"]["stage_2"],
            batch["depth_values"],
            hints,
            validhints,
        )
