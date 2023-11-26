import functools
import numpy as np
from torch import nn


class Discriminator(nn.Module):

    def __init__(self, n_channel=3, n_filters=64, n_layers=3, norm_layer_type='bn', use_sigmoid=True):
        super(Discriminator, self).__init__()

        if norm_layer_type.lower() == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
            use_bias = True
        elif norm_layer_type.lower() == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
            use_bias = False
        else:
            raise ValueError("norm_layer must be either \'instance\' or \'batch\'")

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1) / 2))
        layers = [
            nn.Conv2d(n_channel, n_filters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        mult = 1
        for n in range(1, n_layers):
            mult_prev = mult
            mult = min(2**n, 8)
            layers.append(nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                          kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias))
            layers.append(norm_layer(n_filters * mult))
            layers.append(nn.LeakyReLU(0.2, True))

        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers.append(nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias))
        layers.append(norm_layer(n_filters * mult))
        layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Conv2d(n_filters * mult, 1,
                                kernel_size=kernel_size, stride=1, padding=padding))

        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiscriminatorTail(nn.Module):

    def __init__(self, mult, n_layers, n_filters=64, norm_layer_type='bn', use_sigmoid=False):
        super(DiscriminatorTail, self).__init__()

        if norm_layer_type.lower() == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
            use_bias = True
        elif norm_layer_type.lower() == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
            use_bias = False
        else:
            raise ValueError("norm_layer must be either \'instance\' or \'batch\'")

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1) / 2))
        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers = [
            nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(n_filters * mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_filters * mult, 1,
                      kernel_size=kernel_size, stride=1, padding=padding)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, n_channel=3, n_filters=64, norm_layer_type='batch'):
        super(MultiScaleDiscriminator, self).__init__()

        if norm_layer_type.lower() == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
            use_bias = True
        elif norm_layer_type.lower() == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
            use_bias = False
        else:
            raise ValueError("norm_layer must be either \'instance\' or \'batch\'")

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1) / 2))
        layers = [
            nn.Conv2d(n_channel, n_filters, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        mult = 1
        for n in range(1, 3):
            mult_prev = mult
            mult = min(2**n, 8)
            layers.append(nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                                    kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias))
            layers.append(norm_layer(n_filters * mult))
            layers.append(nn.LeakyReLU(0.2, True))

        self.scale_one = nn.Sequential(*layers)
        self.first_tail = DiscriminatorTail(mult=mult, n_layers=3)

        mult_prev = 4
        mult = 8
        self.scale_two = nn.Sequential(
            nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                      kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
            norm_layer(n_filters * mult),
            nn.LeakyReLU(0.2, True)
        )
        self.second_tail = DiscriminatorTail(mult=mult, n_layers=4)

        mult_prev = mult
        self.scale_three = nn.Sequential(
            nn.Conv2d(n_filters * mult_prev, n_filters * mult,
                      kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
            norm_layer(n_filters * mult),
            nn.LeakyReLU(0.2, True)
        )
        self.third_tail = DiscriminatorTail(mult=mult, n_layers=5)

    def forward(self, x):
        x = self.scale_one(x)
        x_1 = self.first_tail(x)
        x = self.scale_two(x)
        x_2 = self.second_tail(x)
        x = self.scale_three(x)
        x = self.third_tail(x)
        return [x_1, x_2, x]
