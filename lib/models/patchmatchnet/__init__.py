from types import SimpleNamespace

import torch
import torch.nn as nn

from .net import PatchmatchNet, patchmatchnet_loss

__all__ = ["PatchmatchNet", "patchmatchnet_loss", "NetBuilder"]


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
