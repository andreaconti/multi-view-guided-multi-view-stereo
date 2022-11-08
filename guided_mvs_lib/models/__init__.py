from types import SimpleNamespace
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from . import cas_mvsnet, d2hc_rmvsnet, mvsnet, patchmatchnet, ucsnet

__all__ = ["MVSModel", "build_network"]

_NETWORKS = {
    "mvsnet": mvsnet.NetBuilder,
    "cas_mvsnet": cas_mvsnet.NetBuilder,
    "ucsnet": ucsnet.NetBuilder,
    "d2hc_rmvsnet": d2hc_rmvsnet.NetBuilder,
    "patchmatchnet": patchmatchnet.NetBuilder,
}


def build_network(name: str, args: SimpleNamespace):
    try:
        return _NETWORKS[name](args)
    except KeyError:
        raise ValueError("network name in {}".format(", ".join(_NETWORKS.keys())))


class MVSModel(LightningModule):
    def __init__(
        self,
        *,
        args: SimpleNamespace,
        mlflow_run_id: Optional[str] = None,
        v_num: Optional[str] = None,  # (experiment version to show in tqdm)
    ):
        super().__init__()

        # instance the used model
        self.model = build_network(args.model, args)
        self.mlflow_run_id = mlflow_run_id
        self.loss_fn = self.model.loss

        # save train parameters
        hparams = dict(
            model=args.model,
            **{name: value for name, value in args.train.__dict__.items()},
        )
        if hasattr(self.model, "hparams"):
            hparams = dict(**hparams, **self.model.hparams)

        self.save_hyperparameters(hparams)

        # utils
        self._is_val = False
        self.v_num = v_num

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        progress_bar_dict = super().get_progress_bar_dict()
        progress_bar_dict["v_num"] = self.v_num
        return progress_bar_dict

    def forward(self, batch: dict):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.epochs_lr_decay is not None:
            scheduler = MultiStepLR(
                optimizer,
                self.hparams.epochs_lr_decay,
                gamma=self.hparams.epochs_lr_gamma,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _compute_masks(self, depth_dict: Dict) -> Dict:
        return {k: (v > 0).to(torch.float32) for k, v in depth_dict.items()}

    def training_step(self, batch, _):
        self._is_val = False

        outputs = self.model(batch)
        loss = self.loss_fn(
            outputs["loss_data"], batch["depth"], self._compute_masks(batch["depth"])
        )
        self._log_metrics(batch, outputs, loss, "train")

        return loss

    def validation_step(self, batch, _):
        outputs = self.model(batch)
        loss = self.loss_fn(
            outputs["loss_data"], batch["depth"], self._compute_masks(batch["depth"])
        )
        self._log_metrics(batch, outputs, loss, "val")

        return loss

    def _log_metrics(self, batch, outputs, loss, stage):
        on_epoch = stage != "train"

        depth_gt = batch["depth"]["stage_0"]
        mask = (depth_gt > 0).to(torch.float32)
        depth_est = outputs["depth"]["stage_0"]

        # log scalar metrics
        self.log(f"{stage}/loss", loss, on_epoch=on_epoch, on_step=not on_epoch)
        self.log(
            f"{stage}/mae",
            torch.mean(torch.abs(depth_est[mask > 0.5] - depth_gt[mask > 0.5])),
            on_epoch=on_epoch,
            on_step=not on_epoch,
        )
        for thresh in [1, 2, 3, 4, 8]:
            self.log(
                f"{stage}/perc_l1_upper_thresh_{thresh}",
                torch.mean(
                    (torch.abs(depth_est[mask > 0.5] - depth_gt[mask > 0.5]) > thresh).float()
                ),
                on_epoch=on_epoch,
                on_step=not on_epoch,
            )

        # logs only the first image for val, one every 5000 steps for train
        if (stage == "train" and self.global_step % 5000 == 0) or (
            stage == "val" and not self._is_val
        ):
            if stage == "val":
                self._is_val = True

            if self.mlflow_run_id != None:
                for name, map_ in [("depth_pred", depth_est), ("depth_gt", depth_gt)]:

                    fig = plt.figure(tight_layout=True)
                    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(
                        (map_[0] * mask[0]).permute(1, 2, 0).detach().cpu().numpy(), cmap="magma_r"
                    )
                    self.logger.experiment.log_figure(
                        self.mlflow_run_id, fig, f"{stage}/{name}-{self.global_step}.jpg"
                    )
                    plt.close(fig)

                self.logger.experiment.log_image(
                    self.mlflow_run_id,
                    batch["imgs"]["stage_0"][0, 0].permute(1, 2, 0).cpu().numpy(),
                    f"{stage}/image-{self.global_step}.jpg",
                )
