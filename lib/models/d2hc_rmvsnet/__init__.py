import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from .drmvsnet import D2HCRMVSNet
from .vamvsnet import mvsnet_loss

__all__ = ["mvsnet_loss", "D2HCRMVSNet", "NetBuilder"]

_logger = logging.getLogger(__name__)


_DATASETS_SIZE = {
    "dtu_yao": (512, 640),
    "blended_mvs": (576, 768),
    "blended_mvg": (576, 768),
}


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
