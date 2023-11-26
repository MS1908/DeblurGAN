from .generator import Generator
from .discriminator import Discriminator, MultiScaleDiscriminator


def generator_factory(model_config):
    return Generator(**model_config)


def discriminator_factory(model_config):
    discriminator_type = model_config['type']

    if discriminator_type == 'no_gan':
        discriminator = None

    elif discriminator_type == 'patch_gan':
        discriminator = Discriminator(
            n_layers=model_config['n_layers'],
            norm_layer_type=model_config['norm_layer_type'],
            use_sigmoid=False
        )

    elif discriminator_type == 'double_gan':
        patch_d = Discriminator(
            n_layers=model_config['n_layers'],
            norm_layer_type=model_config['norm_layer_type'],
            use_sigmoid=False
        )
        full_d = Discriminator(
            n_layers=5,
            norm_layer_type=model_config['norm_layer_type'],
            use_sigmoid=False
        )
        discriminator = {
            'patch': patch_d, 'full': full_d
        }

    elif discriminator_type == 'multi_scale':
        discriminator = MultiScaleDiscriminator(norm_layer_type=model_config['norm_layer_type'])

    else:
        raise NotImplementedError(f"Discriminator type {discriminator_type} is not implemented")

    return discriminator


def gan_factory(model_config):
    g_config = model_config['generator']
    d_config = model_config['discriminator']

    return generator_factory(g_config), discriminator_factory(d_config)
