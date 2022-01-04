from torch.optim.lr_scheduler import ExponentialLR
from .scheduler import OptimizerScheduler


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


import torch


class NoamOpt(OptimizerScheduler):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, clip, step=0):
        super(NoamOpt, self).__init__(optimizer, None, clip, step)
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, require_zero_grad=False):
        "Update parameters and rate"
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        if require_zero_grad:
            self.optimizer_zero_grad()
        self._update_step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min((step + 1) ** (-0.5), (step + 1) * self.warmup ** (-1.5)))



