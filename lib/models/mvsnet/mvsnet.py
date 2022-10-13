import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import *


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def _split_pad(self, pad):
        if pad % 2 == 0:
            return pad // 2, pad // 2
        else:
            pad_1 = pad // 2
            pad_2 = (pad // 2) + 1
            return pad_1, pad_2

    def _generate_slice(self, pad):
        if pad == 0:
            return slice(0, None)
        elif pad % 2 == 0:
            return slice(pad // 2, -pad // 2)
        else:
            pad_1 = pad // 2
            pad_2 = (pad // 2) + 1
            return slice(pad_1, -pad_2)

    def _pad_to_div_by(self, x, *, div_by=8):
        _, _, _, h, w = x.shape
        new_h = int(np.ceil(h / div_by)) * div_by
        new_w = int(np.ceil(w / div_by)) * div_by
        pad_h_l, pad_h_r = self._split_pad(new_h - h)
        pad_w_t, pad_w_b = self._split_pad(new_w - w)
        return F.pad(x, (pad_w_t, pad_w_b, pad_h_l, pad_h_r))

    def forward(self, x):

        # padding
        _, _, _, h, w = x.shape
        x = self._pad_to_div_by(x, div_by=8)
        _, _, _, new_h, new_w = x.shape

        # regularization
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        # unpadding
        slice_h = self._generate_slice(new_h - h)
        slice_w = self._generate_slice(new_w - w)
        x = x[..., slice_h, slice_w]

        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine
        # self.depth_interval = depth_interval

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values, hints=None, validhints=None):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(
            proj_matrices
        ), "Different number of images and projection matrices"
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(
                    2
                )  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # TODO cost-volume modulation
        if hints is not None and validhints is not None:
            batch_size, feats, height, width = ref_feature.shape
            GAUSSIAN_HEIGHT = 10.0
            GAUSSIAN_WIDTH = 1.0

            # image features are one fourth the original size: subsample the hints and divide them by four
            hints = hints
            hints = F.interpolate(hints, scale_factor=0.25, mode="nearest").unsqueeze(1)
            validhints = validhints
            validhints = F.interpolate(validhints, scale_factor=0.25, mode="nearest").unsqueeze(1)
            hints = hints * validhints

            # add feature and disparity dimensions to hints and validhints
            # and repeat their values along those dimensions, to obtain the same size as cost
            hints = hints.expand(-1, feats, num_depth, -1, -1)
            validhints = validhints.expand(-1, feats, num_depth, -1, -1)

            # create a tensor of the same size as cost, with disparities
            # between 0 and num_disp-1 along the disparity dimension
            depth_hyps = (
                depth_values.unsqueeze(1)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(batch_size, feats, -1, height, width)
                .detach()
            )
            volume_variance = volume_variance * (
                (1 - validhints)
                + validhints
                * GAUSSIAN_HEIGHT
                * (1 - torch.exp(-((depth_hyps - hints) ** 2) / (2 * GAUSSIAN_WIDTH ** 2)))
            )

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(-cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values).unsqueeze(1)
        depth = F.interpolate(depth, scale_factor=4, mode="bilinear")

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
                depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float),
            ).long()
            photometric_confidence = torch.gather(
                prob_volume_sum4, 1, depth_index.unsqueeze(1)
            ).squeeze(1)
            photometric_confidence = F.interpolate(
                photometric_confidence[None], scale_factor=4, mode="bilinear"
            )[0]

        depth_dict = {}
        depth_dict["stage_0"] = depth

        # step 4. depth map refinement
        if not self.refine:
            return {
                "depth": depth_dict,
                "photometric_confidence": photometric_confidence,
                "loss_data": depth_dict,
            }
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {
                "depth": refined_depth,
                "photometric_confidence": photometric_confidence,
                "loss_data": refined_depth,
            }


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask["stage_0"] > 0.5
    return F.smooth_l1_loss(
        depth_est["stage_0"][mask], depth_gt["stage_0"][mask], size_average=True
    )
