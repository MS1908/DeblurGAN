import functools
import timm
import torch
import torch.nn.functional as F
from torch import nn

from .utils import calculate_feature_filters


class FPNHead(nn.Module):

    def __init__(self, num_in, num_mid, num_out):
        super(FPNHead, self).__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNBase(nn.Module):

    def __init__(self, arch, norm_layer, num_filters=128, pretrained=True):
        super(FPNBase, self).__init__()
        self.feature_extractor = timm.create_model(arch, pretrained=pretrained, features_only=True)

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        feature_filters = calculate_feature_filters(arch)

        self.n_features = len(feature_filters)
        assert self.n_features == 5, "Need exactly 5 feature layers for FPN"

        self.lateral4 = nn.Conv2d(feature_filters[4], num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(feature_filters[3], num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(feature_filters[2], num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(feature_filters[1], num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(feature_filters[0], num_filters // 2, kernel_size=1, bias=False)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def encode(self, x):
        # Bottom-up pathway
        enc0, enc1, enc2, enc3, enc4 = self.feature_extractor(x)
        return enc0, enc1, enc2, enc3, enc4

    def decode(self, enc0, enc1, enc2, enc3, enc4):
        # Lateral connections
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return lateral0, map1, map2, map3, map4

    def forward(self, x):
        enc0, enc1, enc2, enc3, enc4 = self.encode(x)
        return self.decode(enc0, enc1, enc2, enc3, enc4)


class FPNInception(FPNBase):

    def __init__(self, arch, norm_layer, num_filters=128, pretrained=True):
        super().__init__(arch, norm_layer, num_filters, pretrained)
        self.pad = nn.ReflectionPad2d(1)

    def decode(self, enc0, enc1, enc2, enc3, enc4):
        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        pad = (1, 2, 1, 2)  # pad last dim by 1 on each side
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode='nearest'))
        map2 = self.td2(F.pad(lateral2, pad, 'reflect') + nn.functional.upsample(map3, scale_factor=2, mode='nearest'))
        map1 = self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode='nearest'))
        return F.pad(lateral0, pad1, 'reflect'), map1, map2, map3, map4


class Generator(nn.Module):

    def __init__(self, arch, norm_layer_type='instance',
                 output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super(Generator, self).__init__()

        if norm_layer_type.lower() == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        elif norm_layer_type.lower() == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        else:
            raise ValueError("norm_layer must be either \'instance\' or \'batch\'")

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        if 'inception' in arch.lower():
            self.fpn = FPNInception(arch=arch, num_filters=num_filters_fpn,
                                    norm_layer=norm_layer, pretrained=pretrained)
        else:
            self.fpn = FPNBase(arch=arch, num_filters=num_filters_fpn,
                               norm_layer=norm_layer, pretrained=pretrained)

        # The segmentation heads on top of the FPN
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)