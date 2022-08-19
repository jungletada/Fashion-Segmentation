import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, dataset='ModaNet', batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.batch_average = batch_average
        self.cuda = cuda

        if dataset == 'ModaNet':
            # self.weight = torch.load('utils/ModaNet_class_weights.pth')
            self.weight = None
            # print(self.weight)
        elif dataset == 'DeepFashion2':
            self.weight = None
        else:
            self.weight = None

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=None):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss



def build_critierion(cuda=True, mode='focal'):
    segloss = SegmentationLosses(cuda=cuda)
    critierion = segloss.build_loss(mode=mode)
    return critierion


def get_loss_aux(criterion, out_dict, target):
    loss1 = criterion(out_dict['output'], target)
    loss2 = criterion(out_dict['aux'], target)
    loss = loss1 + 0.4 * loss2
    return loss1, loss2, loss


def get_loss_NO_aux(criterion, out_dict, target):
    loss = criterion(out_dict['output'], target)
    return loss


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True, dataset='ModaNet')
    a = torch.rand(1, 14, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=1, alpha=0.5).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
