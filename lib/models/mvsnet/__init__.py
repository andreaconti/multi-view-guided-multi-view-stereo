from types import SimpleNamespace

import torch
import torch.nn as nn

from .mvsnet import MVSNet, mvsnet_loss

__all__ = ["mvsnet_loss", "MVSNet", "NetBuilder"]


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
