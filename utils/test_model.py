# -*- coding: utf-8 -*-
import numpy as np
import torch


def test_mode(model, L, mode=0, refield=32, min_size=224, sf=1, modulo=1):
    '''
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Some testing modes
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # (0) normal: test(model, L)
    # (1) pad: test_pad(model, L, modulo=16)
    # (2) split: test_split(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # (3) split only once: test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1)
    # ---------------------------------------
    '''
    if mode == 0:
        E = test(model, L)
    elif mode == 1:
        E = test_pad(model, L, modulo)
    elif mode == 2:
        E = test_split(model, L, refield, min_size, sf, modulo)
    elif mode == 3:
        E = test_onesplit(model, L, refield, min_size, sf, modulo)
    return E


# normal (0)
def test(model, L):
    E = model(L)
    return E['output']


# pad (1)
def test_pad(model, L, modulo=16):
    h, w = L.size()[-2:]
    paddingBottom = int(np.ceil(h/modulo)*modulo-h)
    paddingRight = int(np.ceil(w/modulo)*modulo-w)
    ReplicationPad2d = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))
    L = ReplicationPad2d(L)
    E = model(L)['output']
    E = E[..., :h, :w]
    return E


# split (function)
def test_split_fn(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_size x min_size image, e.g., 256 x 256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]
    if h * w <= min_size**2:
        ReplicationPad2d = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))
        L = ReplicationPad2d(L)
        E = model(L)['output']
        E = E[..., :h*sf, :w*sf]
        # print("h x w: {} x {}".format(h, w))
        # print("h * w <= min_size**2, ReplicationPad2d")
    else:
        top = slice(0, (h//2//refield+1)*refield)
        bottom = slice(h - (h//2//refield+1)*refield, h)
        left = slice(0, (w//2//refield+1)*refield)
        right = slice(w - (w//2//refield+1)*refield, w)

        Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
        # print("h * w > min_size**2, slice")
        if h * w <= 4*(min_size**2):
            Es = [model(Ls[i])['output'] for i in range(4)]
        else:
            Es = [test_split_fn(model, Ls[i], refield=refield, min_size=min_size, sf=sf, modulo=modulo) for i in range(4)]

        b, c = Es[0].size()[:2]
        E = torch.zeros(b, c, sf * h, sf * w).type_as(L)

        E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
        E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
        E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
        E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E


# split (2)
def test_split(model, L, refield=32, min_size=256, sf=1, modulo=1):
    E = test_split_fn(model, L, refield=refield, min_size=min_size, sf=sf, modulo=modulo)
    return E


# split only once (5)
def test_onesplit(model, L, refield=32, min_size=256, sf=1, modulo=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    min_size: min_sizeXmin_size image, e.g., 256 X 256 image
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h//2//refield+1)*refield)
    bottom = slice(h - (h//2//refield+1)*refield, h)
    left = slice(0, (w//2//refield+1)*refield)
    right = slice(w - (w//2//refield+1)*refield, w)

    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = []

    for i in range(4):
        e = model(Ls[i])['output']
        Es.append(e)

    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h//2*sf, :w//2*sf] = Es[0][..., :h//2*sf, :w//2*sf]
    E[..., :h//2*sf, w//2*sf:w*sf] = Es[1][..., :h//2*sf, (-w + w//2)*sf:]
    E[..., h//2*sf:h*sf, :w//2*sf] = Es[2][..., (-h + h//2)*sf:, :w//2*sf]
    E[..., h//2*sf:h*sf, w//2*sf:w*sf] = Es[3][..., (-h + h//2)*sf:, (-w + w//2)*sf:]
    return E


# print model
def print_model(model):
    msg = describe_model(model)
    print(msg)


# print params
def print_params(model):
    msg = describe_params(model)
    print(msg)


# model information
def info_model(model):
    msg = describe_model(model)
    return msg


# params information
def info_params(model):
    msg = describe_params(model)
    return msg


# model name and total number of parameters
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# parameters description
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg


if __name__ == '__main__':
    class Net(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=3):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            return {'output': x}

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    model = Net()
    model = model.eval()
    x = torch.randn((2,3,600,400))
    torch.cuda.empty_cache()
    with torch.no_grad():
        for mode in range(4):
            print("mode: {}".format(mode), end=', ')
            y = test_mode(model, x, mode)
            print(y.shape)
