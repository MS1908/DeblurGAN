import math
from torch.optim import lr_scheduler


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max=30, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


class LinearDecay(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, start_epoch=0, min_lr=0, last_epoch=-1):
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        return [base_lr - ((base_lr - self.min_lr) / self.num_epochs) * (self.last_epoch - self.start_epoch) for
                base_lr in self.base_lrs]
