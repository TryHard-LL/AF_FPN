'''
Description:
Author：LL-Version-V1
Date: 2023-08-17
LastEditTime: 2023-08-22
Description:AF_FPN（AAM+FEM）
Original Prper：https://arxiv.org/abs/2112.08782
'''
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import ConvModule
from init_weights import xavier_init
# from ..module.conv import ConvModule
# from ..module.init_weights import xavier_init

class AAM(nn.Module):
    def __init__(self, feature_map_shape, pool_nums=3, in_channels=704, out_channels=128):
        super(AAM, self).__init__()
        self.pool_nums = pool_nums
        self.out_channels = out_channels
        self.adaptive_average_pool = nn.ModuleList()
        self.cv1 = nn.ModuleList()
        self.M5_feature_map_shape = feature_map_shape
        for i in range(self.pool_nums):
            self.adaptive_average_pool.append(nn.AdaptiveAvgPool2d(int(random.uniform(0.1, 0.5) * self.M5_feature_map_shape)))
            self.cv1.append((nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)))
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.cv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        upsample_out = []


        for i in range(self.pool_nums):
            # adaptive average pooling
            pool = self.adaptive_average_pool[i](x)

            # 1 * 1 conv to obtain the same channel dimension 256
            cv1 = self.cv1[i](pool)

            # upsamling
            upsample = F.interpolate(cv1, size=[x.size(2), x.size(3)], mode="nearest")
            upsample_out.append(upsample)

        # concat feature map
        cat_out = torch.cat((upsample_out[0], upsample_out[1], upsample_out[2]), dim=1)

        # 1 * 1 ——> ReLU ——> 3 * 3 ——> sigmoid ——> spatial weight map
        weight_map = self.layer(cat_out)
        out = cat_out * weight_map

        out = torch.split(out, dim=1, split_size_or_sections=self.out_channels)

        # M6
        out = sum(out)
        cv2 = self.cv2(x)
        out = out + cv2

        return out

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedConvBlock, self).__init__()
        # 使用与膨胀因子相同的padding因子，即可确保输入、输出特征图大小一致。
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.dilated_conv(x))
        return out

class MultiDilatedConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(MultiDilatedConvModel, self).__init__()
        self.conv_blocks = nn.ModuleList()

        for dilation in dilations:
            self.conv_blocks.append(DilatedConvBlock(in_channels, out_channels, dilation))

    def forward(self, x):
        out = []
        for conv_block in self.conv_blocks:
            out.append(conv_block(x))
        return out


class FEM(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 3, 5]):
        super(FEM, self).__init__()
        self.multi_dilation = MultiDilatedConvModel(in_channels, out_channels, dilations)

    def forward(self, x):
        out = self.multi_dilation(x)
        return tuple(out)

class AF_FPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs, image_size=640,
                 start_level=0, end_level=-1, conv_cfg=None, norm_cfg=None,
                 activation=None):
        super(AF_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, activation=activation, inplace=False)

            self.lateral_convs.append(l_conv)

        # AAM
        self.aam = AAM(feature_map_shape=image_size // 16, in_channels=in_channels[-1], out_channels=out_channels)

        # FEM
        self.fem = FEM(in_channels=out_channels, out_channels=out_channels)
        self.avp = nn.ModuleList()
        self.avp.append(nn.AdaptiveAvgPool2d(image_size // 4))
        self.avp.append(nn.AdaptiveAvgPool2d(image_size // 8))
        self.avp.append(nn.AdaptiveAvgPool2d(image_size // 16))

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        M6 = self.aam(inputs[-1])
        laterals[-1] = M6 + laterals[-1]

        # Single FEM Output
        laterals_0 = self.fem(laterals[0])
        laterals_1 = self.fem(laterals[1])
        laterals_2 = self.fem(laterals[2])

        # Add FEM-Output
        laterals[0] = laterals[0] + laterals_0[0] + laterals_0[1] + laterals_0[2]
        laterals[1] = laterals[1] + laterals_1[0] + laterals_1[1] + laterals_1[2]
        laterals[2] = laterals[2] + laterals_2[0] + laterals_2[1] + laterals_2[2]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode="bilinear")

        # build outputs
        outs = [
            # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            laterals[i] for i in range(used_backbone_levels)
        ]

        return tuple(outs)


if __name__ == '__main__':
    in_channels=[176, 352, 704]
    out_channels=128
    num_outs=5
    activation='LeakyReLU'

    af_fpn = AF_FPN(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs, activation=activation)
    torch.save(af_fpn.state_dict(), "af_fpn.ckpt")

    input = (torch.randn((1, in_channels[0], 160, 160)),
             torch.randn((1, in_channels[1], 80, 80)),
             torch.randn((1, in_channels[2], 40, 40))
             )
    output = af_fpn(input)

    print(f"len(output)：{len(output)}")
    # 以下输出的特征图大小为原特征图大小的1/8、1/16与1/32。
    print(f"output[0].shape：{output[0].shape} || output[1].shape:{output[1].shape} || output[2].shape:{output[2].shape}")

