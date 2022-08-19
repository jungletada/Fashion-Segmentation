##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0):
        self.mode = mode
        self.lr = base_lr
        # print('Using {} LR Scheduler!'.format(self.mode))
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


def create_lr_scheduler(optimizer,
                        num_step,
                        epochs,
                        warmup=False,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        power=0.9,
                        last_epoch=0):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** power
    if last_epoch == -1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=-1)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=last_epoch*num_step)


def create_poly_lr_scheduler(
                        optimizer,
                        num_step,
                        epochs,
                        warmup=True,
                        endwarmup=True,
                        warmup_epochs=1,
                        end_warmup_epochs=1,
                        warmup_factor=1e-3,
                        power=0.9,
                        last_epoch=0):
    assert num_step > 0 and epochs > 0

    if warmup is False:
        warmup_epochs = 0
    if endwarmup is False:
        end_warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        x: 当前迭代的step
        """
        if warmup and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        elif end_warmup_epochs and x <= (epochs - end_warmup_epochs) * num_step:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** power
        else:
            alpha = float(x - (epochs - end_warmup_epochs) * num_step) / (end_warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return alpha * 1e-3

    if last_epoch == -1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=-1)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=last_epoch*num_step)


def create_MultiSetpLR(optimizer, epochs:int, last_epoch=-1):
    if epochs == 200:
        milestones = [40, 80, 120, 140, 160, 180, 190]
    elif epochs == 110:
        milestones = [20, 40, 60, 80, 90, 100]
    else:
        milestones = []
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.8, last_epoch=last_epoch)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from models.Segmentor.deeplab import DeepLab
    lr_list = []
    initial_lr = 1e-3
    epochs = 120
    model = DeepLab()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(),
         'initial_lr':initial_lr}],
        lr=initial_lr,
        betas=(0.9, 0.99),
        weight_decay=5e-2)

    lr_scheduler = create_poly_lr_scheduler(
        optimizer,
        num_step=1508,
        epochs=epochs,
        warmup=True,
        endwarmup=True,
        warmup_epochs=1,
        end_warmup_epochs=1,
        warmup_factor=1e-3,
        power=0.9,
        last_epoch=-1)

    for _ in range(epochs):
        for _ in range(1508):
            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            lr_list.append(lr)

    plt.plot(range(len(lr_list[-1508:])), lr_list[-1508:])
    plt.show()
    # print(lr_list[1508 * 60])
    # for _ in range(epochs):
    #     multi_lr_scheduler.step()
    #     multi_lr = optimizer.param_groups[0]["lr"]
    #     lr_list.append(multi_lr)
    #
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

