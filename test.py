import torch
import yaml

from models import gan_factory


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/exp.yaml', 'r'))
    G, D = gan_factory(config)

    dummy_input = torch.randn((1, 3, 1024, 512))
    dummy_input = dummy_input
    g_output = G(dummy_input)
    d_output = D['full'](dummy_input)

    print('generator output size', g_output.size())
    print('discriminator output size', d_output.size())
