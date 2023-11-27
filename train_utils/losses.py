import torch
from torch import nn
from torchvision import models, transforms

from misc_utils import ImagePool


class SimpleLoss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, fake_img, real_img):
        return self.criterion(fake_img, real_img)


class PerceptualLoss(SimpleLoss):
    def __init__(self, criterion, device):
        super().__init__(criterion)

        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            cnn = models.vgg19(pretrained=True).features
            cnn.to(device)

            self.loss_model = nn.Sequential()
            self.loss_model.to(device)
            self.loss_model.eval()
            for i, layer in enumerate(list(cnn)):
                self.loss_model.add_module(str(i), layer)
                if i == 14:
                    break

    def __call__(self, fake_img, real_img):
        fake_img = (fake_img + 1) / 2.
        real_img = (real_img + 1) / 2.

        fake_img[0, :, :, :] = self.transform(fake_img[0, :, :, :])
        real_img[0, :, :, :] = self.transform(real_img[0, :, :, :])

        f_fake = self.loss_model.forward(fake_img)
        f_real = self.loss_model.forward(real_img)
        f_real_no_grad = f_real.detach()

        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fake_img, real_img)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 device=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_tensor = None
        self.fake_tensor = None
        self.device = device
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, inputs, target_is_real=True):
        if target_is_real:
            if self.real_tensor is None or self.real_tensor.numel() != inputs.numel():
                self.real_tensor = torch.FloatTensor(inputs.size()).fill_(self.real_label)
            target_tensor = self.real_tensor

        else:
            if self.fake_tensor is None or self.fake_tensor.numel() != inputs.numel():
                self.fake_tensor = torch.FloatTensor(inputs.size()).fill_(self.fake_label)
            target_tensor = self.fake_tensor

        if self.device is not None:
            target_tensor = target_tensor.to(self.device)
        return target_tensor

    def __call__(self, inputs, target_is_real):
        target_tensor = self.get_target_tensor(inputs, target_is_real)
        return self.loss(inputs, target_tensor)


class DiscLoss(nn.Module):
    def __init__(self, device=None):
        super(DiscLoss, self).__init__()
        self.criterion = GANLoss(use_l1=False, device=device)

    def get_g_loss(self, model, fake_b, real_b):
        # First, G(A) should fake the discriminator
        pred_fake = model(fake_b)
        return self.criterion(pred_fake, True)

    def __call__(self, model, fake_b, real_b):
        # Fake
        pred_fake = model.forward(fake_b.detach())  # stop backprop to the generator by detaching fake_B
        loss_D_fake = self.criterion(pred_fake, False)  # should be close to zero

        # Real
        pred_real = model.forward(real_b)
        loss_D_real = self.criterion(pred_real, True)

        return (loss_D_fake + loss_D_real) * 0.5


class DiscLossLS(DiscLoss):
    def __init__(self, device):
        super(DiscLossLS, self).__init__()
        self.criterion = GANLoss(use_l1=True, device=device)


class RelativisticDiscLoss(DiscLoss):
    def __init__(self, device):
        super(RelativisticDiscLoss, self).__init__()

        self.criterion = GANLoss(use_l1=False, device=device)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, model, fake_b, real_b):
        # First, G(A) should fake the discriminator
        pred_fake = model.forward(fake_b)

        # Real
        pred_real = model.forward(real_b)
        loss = (self.criterion(pred_real - torch.mean(self.fake_pool.query()), True)
                + self.criterion(pred_fake - torch.mean(self.real_pool.query()), False)) / 2.

        return loss

    def __call__(self, model, fake_b, real_b):
        # Fake
        pred_fake = model.forward(fake_b.detach())
        self.fake_pool.add(pred_fake)
        loss_D_fake = self.criterion(pred_fake - torch.mean(self.real_pool.query()), False)

        # Real
        pred_real = model.forward(real_b)
        self.real_pool.add(pred_real)
        loss_D_real = self.criterion(pred_real - torch.mean(self.fake_pool.query()), True)

        return (loss_D_fake + loss_D_real) * 0.5


class RelativisticDiscLossLS(DiscLoss):
    def __init__(self, device):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterion = GANLoss(use_l1=True, device=device)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, model, fake_b, real_b):
        pred_fake = model.forward(fake_b)

        pred_real = model.forward(real_b)
        loss = (torch.mean((pred_real - torch.mean(self.fake_pool.query()) + 1)**2)
                + torch.mean((pred_fake - torch.mean(self.real_pool.query()) - 1)**2)) / 2.

        return loss

    def __call__(self, model, fake_b, real_b):
        # Fake
        pred_fake = model.forward(fake_b.detach())
        self.fake_pool.add(pred_fake)

        # Real
        pred_real = model.forward(real_b)
        self.real_pool.add(pred_real)

        loss = (torch.mean((pred_real - torch.mean(self.fake_pool.query()) - 1)**2)
                + torch.mean((pred_fake - torch.mean(self.real_pool.query()) + 1)**2)) / 2.

        return loss


class DiscLossWGANGP(DiscLossLS):
    def __init__(self, device):
        super(DiscLossWGANGP, self).__init__(device)
        self.L = 10
        self.device = device


def loss_factory(content_loss_type='perceptual', disc_loss_type='ragan-ls', device=None):
    if content_loss_type == 'perceptual':
        content_loss = PerceptualLoss(criterion=nn.MSELoss(), device=device)
    elif content_loss_type == 'l1':
        content_loss = SimpleLoss(criterion=nn.L1Loss())
    else:
        raise NotImplementedError(f"Content loss {content_loss_type} not implemented.")

    if disc_loss_type == 'gan':
        disc_loss = DiscLoss(device=device)
    elif disc_loss_type == 'lsgan':
        disc_loss = DiscLossLS(device=device)
    elif disc_loss_type == 'ragan':
        disc_loss = RelativisticDiscLoss(device=device)
    elif disc_loss_type == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS(device=device)
    else:
        raise NotImplementedError(f"GAN Loss {disc_loss_type} not recognized.")

    return content_loss, disc_loss
