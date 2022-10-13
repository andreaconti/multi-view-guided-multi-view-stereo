from types import SimpleNamespace

import torch
import torch.nn as nn

from .cas_mvsnet import CascadeMVSNet, cas_mvsnet_loss

__all__ = ["cas_mvsnet_loss", "CascadeMVSNet", "NetBuilder"]


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
