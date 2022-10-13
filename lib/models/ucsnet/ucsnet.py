import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules import *


def compute_depth(
    feats,
    proj_mats,
    depth_samps,
    cost_reg,
    lamb,
    is_training=False,
    hints=None,
    validhints=None,
    scale=1,
):
    """
    :param feats: [(B, C, H, W), ] * num_views
    :param proj_mats: [()]
    :param depth_samps:
    :param cost_reg:
    :param lamb:
    :return:
    """

    #    print(proj_mats.shape)
    proj_mats = torch.unbind(proj_mats, 1)
    num_views = len(feats)
    num_depth = depth_samps.shape[1]

    assert len(proj_mats) == num_views, "Different number of images and projection matrices"

    ref_feat, src_feats = feats[0], feats[1:]
    ref_proj, src_projs = proj_mats[0], proj_mats[1:]

    ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    # todo optimize impl
    for src_fea, src_proj in zip(src_feats, src_projs):
        warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_samps)

        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2)  # in_place method
        del warped_volume
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

    # TODO cost-volume modulation
    if hints is not None and validhints is not None:
        batch_size, feats, height, width = ref_feat.shape
        GAUSSIAN_HEIGHT = 10.0
        GAUSSIAN_WIDTH = 1.0

        # image features are one fourth the original size: subsample the hints and divide them by four
        hints = hints
        hints = F.interpolate(hints, scale_factor=1 / scale, mode="nearest").unsqueeze(1)
        validhints = validhints
        validhints = F.interpolate(validhints, scale_factor=1 / scale, mode="nearest").unsqueeze(1)
        hints = hints * validhints

        # add feature and disparity dimensions to hints and validhints
        # and repeat their values along those dimensions, to obtain the same size as cost
        hints = hints.expand(-1, feats, num_depth, -1, -1)
        validhints = validhints.expand(-1, feats, num_depth, -1, -1)

        # create a tensor of the same size as cost, with disparities
        # between 0 and num_disp-1 along the disparity dimension
        depth_hyps = depth_samps.unsqueeze(1).expand(batch_size, feats, -1, height, width).detach()
        volume_variance = volume_variance * (
            (1 - validhints)
            + validhints
            * GAUSSIAN_HEIGHT
            * (1 - torch.exp(-((depth_hyps - hints) ** 2) / (2 * GAUSSIAN_WIDTH ** 2)))
        )

    prob_volume_pre = cost_reg(volume_variance).squeeze(1)
    prob_volume = F.softmax(prob_volume_pre, dim=1)
    depth = depth_regression(prob_volume, depth_values=depth_samps)

    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(
            F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0
        ).squeeze(1)
        depth_index = depth_regression(
            prob_volume,
            depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float),
        ).long()
        depth_index = depth_index.clamp(min=0, max=num_depth - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

    samp_variance = (depth_samps - depth.unsqueeze(1)) ** 2
    exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

    return {"depth": depth, "photometric_confidence": prob_conf, "variance": exp_variance}


class UCSNet(nn.Module):
    def __init__(
        self,
        lamb=1.5,
        stage_configs=[64, 32, 8],
        grad_method="detach",
        base_chs=[8, 8, 8],
        feat_ext_ch=8,
    ):
        super(UCSNet, self).__init__()

        self.stage_configs = stage_configs
        self.grad_method = grad_method
        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(stage_configs)
        self.ds_ratio = {
            "stage_2": 4.0,  # "stage_1": 4.0,
            "stage_1": 2.0,  # "stage_2": 2.0,
            "stage_0": 1.0,  # "stage_3": 1.0
        }

        self.feature_extraction = FeatExtNet(
            base_channels=feat_ext_ch,
            num_stage=self.num_stage,
        )

        self.cost_regularization = nn.ModuleList(
            [
                CostRegNet(
                    in_channels=self.feature_extraction.out_channels[i],
                    base_channels=self.base_chs[i],
                )
                for i in [2, 1, 0]
            ]
        )  # range(self.num_stage)])

    def forward(self, imgs, proj_matrices, depth_values, hints=None, validhints=None):
        features = []
        for nview_idx in range(imgs.shape[1]):
            img = imgs[:, nview_idx]
            features.append(self.feature_extraction(img))

        outputs = {}
        depth, cur_depth, exp_var = None, None, None
        for stage_idx in [2, 1, 0]:  # range(self.num_stage):
            features_stage = [feat["stage_{}".format(stage_idx)] for feat in features]
            proj_matrices_stage = proj_matrices["stage_{}".format(stage_idx)]  # + 1)]
            stage_scale = self.ds_ratio["stage_{}".format(stage_idx)]  # + 1)]
            cur_h = img.shape[2] // int(stage_scale)
            cur_w = img.shape[3] // int(stage_scale)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                else:
                    cur_depth = depth

                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [cur_h, cur_w], mode="bilinear")
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode="bilinear")

            else:
                cur_depth = depth_values

            depth_range_samples = uncertainty_aware_samples(
                cur_depth=cur_depth,
                exp_var=exp_var,
                ndepth=self.stage_configs[2 - stage_idx],
                dtype=img[0].dtype,
                device=img[0].device,
                shape=[img.shape[0], cur_h, cur_w],
            )

            outputs_stage = compute_depth(
                features_stage,
                proj_matrices_stage,
                depth_samps=depth_range_samples,
                cost_reg=self.cost_regularization[stage_idx],
                lamb=self.lamb,
                is_training=self.training,
                hints=hints,
                validhints=validhints,
                scale=stage_scale,
            )

            depth = outputs_stage["depth"]
            exp_var = outputs_stage["variance"]

            outputs["stage_{}".format(stage_idx)] = outputs_stage["depth"].unsqueeze(1)

        return {
            "depth": outputs,
            "loss_data": {
                "depth": outputs,
            },
            "photometric_confidence": outputs_stage["photometric_confidence"],
        }


def ucsnet_loss(outputs, labels, masks, weights=[0.5, 1.0, 2.0]):
    tot_loss = 0
    for stage_id in [0, 1, 2]:  # range(3):
        depth_i = outputs["stage_{}".format(stage_id)]  # .unsqueeze(1)
        label_i = labels["stage_{}".format(stage_id)]
        mask_i = masks["stage_{}".format(stage_id)].bool()
        depth_loss = F.smooth_l1_loss(depth_i[mask_i], label_i[mask_i], reduction="mean")
        tot_loss += depth_loss * weights[2 - stage_id]
    return tot_loss
