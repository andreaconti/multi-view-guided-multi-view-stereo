import logging
import warnings
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .drmvsnet import D2HCRMVSNet
from .vamvsnet import mvsnet_loss

__all__ = ["mvsnet_loss", "D2HCRMVSNet", "NetBuilder", "SimpleInterfaceNet"]

_logger = logging.getLogger(__name__)


_DATASETS_SIZE = {
    "dtu_yao": (512, 640),
    "blended_mvs": (576, 768),
    "blended_mvg": (576, 768),
}


class SimpleInterfaceNet(nn.Module):
    """
    Simple common interface to call the pretrained models
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = D2HCRMVSNet(*args, **kwargs)
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
            proj_mat = extrinsics.clone()
            intrinsics_copy = intrinsics.clone()
            intrinsics_copy[..., :2, :] = intrinsics_copy[..., :2, :] / 4
            proj_mat[..., :3, :4] = intrinsics_copy @ proj_mat[..., :3, :4]

            # image resize
            imgs = torch.stack(
                [
                    F.interpolate(img, scale_factor=0.25, mode="bilinear", align_corners=True)
                    for img in torch.unbind(imgs, 2)
                ],
                2,
            )

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

        self.model = D2HCRMVSNet()
        self.loss = mvsnet_loss

    def forward(self, batch: dict):

        hints, validhints = None, None
        if "hints" in batch:
            hints = batch["hints"]
            validhints = (hints > 0).to(torch.float32)

        out = self.model(
            batch["imgs"]["stage_2"],
            batch["proj_matrices"]["stage_2"],
            batch["depth_values"],
            hints,
            validhints,
        )
        return out
