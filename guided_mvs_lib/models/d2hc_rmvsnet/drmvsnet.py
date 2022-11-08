import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .module import *

# Recurrent Multi-scale Module
from .rnnmodule import *
from .submodule import volumegatelight, volumegatelightgn
from .vamvsnet import *

# More scale feature map submodule
from .vamvsnet_high_submodule import *


class D2HCRMVSNet(MVSNet):
    def __init__(
        self,
        refine=True,
        fea_net="FeatNet",
        cost_net="UNetConvLSTM",
        refine_net="RefineNet",
        origin_size=False,
        cost_aggregation=0,
        dp_ratio=0.0,
        image_scale=0.25,
        max_h=512,
        max_w=640,
        reg_loss=False,
        return_depth=False,
        gn=True,
        pyramid=-1,
    ):
        super(D2HCRMVSNet, self).__init__(
            refine=True,
            fea_net="FeatNet",
            cost_net="UNetConvLSTM",
            refine_net="RefineNet",
            origin_size=False,
            cost_aggregation=0,
            dp_ratio=0.0,
            image_scale=0.25,
        )  # parent init

        self.gn = gn
        self.cost_aggregation = cost_aggregation

        if fea_net == "FeatNet":
            self.feature = FeatNet(gn=self.gn)
            # self.feature = FeatNetGN()
        if cost_net == "UNetConvLSTM":
            ## 3 LSTM layers
            ## Memory Consumption: 1 batch, 7G
            if pyramid == -1:
                input_size = (int(max_h * image_scale), int(max_w * image_scale))  # height, width
            elif pyramid == 0:
                input_size = (296, 400)
            elif pyramid == 1:
                input_size = (144, 200)
            elif pyramid == 2:
                input_size = (72, 96)
            # print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
            input_dim = [32, 16, 16, 32, 32]
            hidden_dim = [16, 16, 16, 16, 8]
            num_layers = 5
            kernel_size = [(3, 3) for i in range(num_layers)]

            self.cost_regularization = UNetConvLSTM(
                input_size,
                input_dim,
                hidden_dim,
                kernel_size,
                num_layers,
                batch_first=False,
                bias=True,
                return_all_layers=False,
                gn=self.gn,
            )
        elif cost_net == "UNetPPConvLSTMV3":
            ## UNet++ skip connection, 3 LSTM layers, using upsample not deconvolution
            ## Memory Consumption: 1 batch, 10577MiB
            if pyramid == -1:
                input_size = (int(max_h * image_scale), int(max_w * image_scale))  # height, width
            elif pyramid == 0:
                input_size = (296, 400)
            elif pyramid == 1:
                input_size = (144, 200)
            elif pyramid == 2:
                input_size = (72, 96)
            # print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
            input_dim = [
                32,
                16,
                16,
                32,
                32,
                48,
            ]  # add mid x0_1, encoder: 1,2,3, decoder:4,5,6 [6 is the last one]
            hidden_dim = [16, 16, 16, 16, 16, 8]
            num_layers = 6
            kernel_size = [(3, 3) for i in range(num_layers)]

            self.cost_regularization = UNetPPConvLSTMV3(
                input_size,
                input_dim,
                hidden_dim,
                kernel_size,
                num_layers,
                batch_first=False,
                bias=True,
                return_all_layers=False,
                gn=self.gn,
            )
        elif cost_net == "UNetPPConvLSTMV3UPS":
            ## UNet++ skip connection, 3 LSTM layers, using upsample not deconvolution
            ## Memory Consumption: 1 batch, 9445MiB
            if pyramid == -1:
                input_size = (int(max_h * image_scale), int(max_w * image_scale))  # height, width
            elif pyramid == 0:
                input_size = (296, 400)
            elif pyramid == 1:
                input_size = (144, 200)
            elif pyramid == 2:
                input_size = (72, 96)
            # print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
            input_dim = [
                32,
                16,
                16,
                32,
                32,
                48,
            ]  # add mid x0_1, encoder: 1,2,3, decoder:4,5,6 [6 is the last one]
            hidden_dim = [16, 16, 16, 16, 16, 8]
            num_layers = 6
            kernel_size = [(3, 3) for i in range(num_layers)]

            self.cost_regularization = UNetPPConvLSTMV3UPS(
                input_size,
                input_dim,
                hidden_dim,
                kernel_size,
                num_layers,
                batch_first=False,
                bias=True,
                return_all_layers=False,
                gn=self.gn,
            )
        elif cost_net == "UNetConvLSTMV4":
            ## 4 LSTM layers
            ## Memory Consumption: 1 batch,
            if pyramid == -1:
                input_size = (int(max_h * image_scale), int(max_w * image_scale))  # height, width
            elif pyramid == 0:
                input_size = (296, 400)
            elif pyramid == 1:
                input_size = (144, 200)
            elif pyramid == 2:
                input_size = (72, 96)
            # print('input UNetConvLSTM H,W: {}, {}'.format(input_size[0], input_size[1]))
            num_layers = 7
            input_dim = [32, 16, 16, 16, 32, 32, 32]
            hidden_dim = [16, 16, 16, 16, 16, 16, 8]
            kernel_size = [(3, 3) for i in range(num_layers)]

            self.cost_regularization = UNetConvLSTMV4(
                input_size,
                input_dim,
                hidden_dim,
                kernel_size,
                num_layers,
                batch_first=False,
                bias=True,
                return_all_layers=False,
                gn=self.gn,
            )

        # Cost Aggregation
        self.gatenet = gatenet(self.gn, 32)

        self.reg_loss = reg_loss
        self.return_depth = return_depth

    def forward(self, imgs, proj_matrices, depth_values, hints=None, validhints=None):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(
            proj_matrices
        ), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # process DrMVSNet

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # Recurrent process i-th depth layer
        # initialization for drmvsnet # recurrent module
        cost_reg_list = []
        hidden_state = None
        if not self.return_depth:  # Training Phase;
            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume

                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(
                        src_fea, src_proj, ref_proj, depth_values[:, d]
                    )
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume)
                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)

                # TODO cost-volume modulation
                if hints is not None and validhints is not None:
                    batch_size, feats, height, width = ref_feature.shape
                    GAUSSIAN_HEIGHT = 10.0
                    GAUSSIAN_WIDTH = 1.0

                    # image features are one fourth the original size: subsample the hints and divide them by four
                    scale = hints.shape[-1] / width
                    curr_hints = F.interpolate(
                        hints, scale_factor=1 / scale, mode="nearest"
                    )  # .unsqueeze(1)
                    curr_validhints = F.interpolate(
                        validhints, scale_factor=1 / scale, mode="nearest"
                    )  # .unsqueeze(1)
                    curr_hints = curr_hints * curr_validhints

                    # add feature and disparity dimensions to hints and validhints
                    # and repeat their values along those dimensions, to obtain the same size as cost
                    curr_hints = curr_hints.expand(-1, feats, -1, -1)
                    curr_validhints = curr_validhints.expand(-1, feats, -1, -1)

                    # create a tensor of the same size as cost, with disparities
                    # between 0 and num_disp-1 along the disparity dimension
                    depth_hyps = (
                        depth_values.unbind(1)[d]
                        .unsqueeze(1)
                        .expand(batch_size, feats, height, width)
                        .detach()
                    )
                    volume_variance = volume_variance * (
                        (1 - curr_validhints)
                        + curr_validhints
                        * GAUSSIAN_HEIGHT
                        * (
                            1
                            - torch.exp(
                                -((depth_hyps - curr_hints) ** 2) / (2 * GAUSSIAN_WIDTH ** 2)
                            )
                        )
                    )

                # step 3. cost volume regularization
                cost_reg, hidden_state = self.cost_regularization(
                    -1 * volume_variance, hidden_state, d
                )
                cost_reg_list.append(cost_reg)

            prob_volume = torch.stack(cost_reg_list, dim=1).squeeze(2)
            prob_volume = F.softmax(
                prob_volume, dim=1
            )  # get prob volume use for recurrent to decrease memory consumption

            depth = depth_regression(prob_volume, depth_values=depth_values).unsqueeze(1)
            depth = F.interpolate(depth, scale_factor=4, mode="bilinear")

            depth_dict = {}
            depth_dict["stage_0"] = depth

            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = (
                    4
                    * F.avg_pool3d(
                        F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)),
                        (4, 1, 1),
                        stride=1,
                        padding=0,
                    ).squeeze(1)
                )
                depth_index = depth_regression(
                    prob_volume,
                    depth_values=torch.arange(
                        num_depth, device=prob_volume.device, dtype=torch.float
                    ),
                ).long()
                photometric_confidence = torch.gather(
                    prob_volume_sum4, 1, depth_index.unsqueeze(1)
                )
                photometric_confidence = F.interpolate(
                    photometric_confidence, scale_factor=4, mode="bilinear"
                )[0]

            return {
                "depth": depth_dict,
                "loss_data": depth_dict,
                "photometric_confidence": photometric_confidence,
            }
        else:
            shape = ref_feature.shape
            depth_image = torch.zeros(shape[0], shape[2], shape[3]).to(ref_feature.device)
            max_prob_image = torch.zeros(shape[0], shape[2], shape[3]).to(ref_feature.device)
            exp_sum = torch.zeros(shape[0], shape[2], shape[3]).to(ref_feature.device)

            for d in range(num_depth):
                # step 2. differentiable homograph, build cost volume

                ref_volume = ref_feature
                warped_volumes = None
                for src_fea, src_proj in zip(src_features, src_projs):
                    warped_volume = homo_warping_depthwise(
                        src_fea, src_proj, ref_proj, depth_values[:, d]
                    )
                    warped_volume = (warped_volume - ref_volume).pow_(2)
                    reweight = self.gatenet(warped_volume)  # saliency
                    if warped_volumes is None:
                        warped_volumes = (reweight + 1) * warped_volume
                    else:
                        warped_volumes = warped_volumes + (reweight + 1) * warped_volume
                volume_variance = warped_volumes / len(src_features)

                # TODO cost-volume modulation
                if hints is not None and validhints is not None:
                    batch_size, feats, height, width = ref_feature.shape
                    GAUSSIAN_HEIGHT = 10.0
                    GAUSSIAN_WIDTH = 1.0

                    # image features are one fourth the original size: subsample the hints and divide them by four
                    scale = hints.shape[-1] / width
                    curr_hints = F.interpolate(
                        hints, scale_factor=1 / scale, mode="nearest"
                    )  # .unsqueeze(1)
                    curr_validhints = F.interpolate(
                        validhints, scale_factor=1 / scale, mode="nearest"
                    )  # .unsqueeze(1)
                    curr_hints = curr_hints * curr_validhints

                    # add feature and disparity dimensions to hints and validhints
                    # and repeat their values along those dimensions, to obtain the same size as cost
                    curr_hints = curr_hints.expand(-1, feats, -1, -1)
                    curr_validhints = curr_validhints.expand(-1, feats, -1, -1)

                    # create a tensor of the same size as cost, with disparities
                    # between 0 and num_disp-1 along the disparity dimension
                    depth_hyps = (
                        depth_values.unbind(1)[d]
                        .unsqueeze(1)
                        .expand(batch_size, feats, height, width)
                        .detach()
                    )
                    volume_variance = volume_variance * (
                        (1 - curr_validhints)
                        + curr_validhints
                        * GAUSSIAN_HEIGHT
                        * (
                            1
                            - torch.exp(
                                -((depth_hyps - curr_hints) ** 2) / (2 * GAUSSIAN_WIDTH ** 2)
                            )
                        )
                    )

                # step 3. cost volume regularization
                cost_reg, hidden_state = self.cost_regularization(
                    -1 * volume_variance, hidden_state, d
                )

                # Start to caculate depth index
                # print('cost_reg: ', cost_reg.shape())
                prob = torch.exp(cost_reg.squeeze(1))

                d_idx = d
                depth = depth_values[:, d]  # B
                temp_depth_image = depth.view(shape[0], 1, 1).repeat(1, shape[2], shape[3])
                update_flag_image = (max_prob_image < prob).type(torch.float)
                # print('update num: ', torch.sum(update_flag_image))
                new_max_prob_image = torch.mul(update_flag_image, prob) + torch.mul(
                    1 - update_flag_image, max_prob_image
                )
                new_depth_image = torch.mul(update_flag_image, temp_depth_image) + torch.mul(
                    1 - update_flag_image, depth_image
                )
                max_prob_image = new_max_prob_image
                depth_image = new_depth_image
                exp_sum = exp_sum + prob

            forward_exp_sum = exp_sum
            forward_depth_map = depth_image

            return {
                "depth": forward_depth_map,
                "photometric_confidence": max_prob_image / forward_exp_sum,
            }
