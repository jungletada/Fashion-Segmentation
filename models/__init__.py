from models.backbone.SSA import shunted_t, shunted_s, shunted_b
from models.backbone.fashion0317 import Fashion_t,Fashion_s,Fashion_b
from models.Segmentor.swin_transformer import swin
from models.Segmentor.deeplab import DeepLab
from models.Segmentor.pspnet import PSPNet
from models.Segmentor.unet import UNet
from models.Segmentor.danet import DANet
from models.Segmentor.segformer import SegFormer
from models.Segmentor.semask_swin_transformer import SeMask

version = '-03-17'

def build_model(modelname='SegFormer_b0', decode='Upernet'):
    if modelname == 'shunted_b':
        return {'modelname': 'shunted_b',
                'net': shunted_b(decode)}
    elif modelname == 'shunted_s':
        return {'modelname': 'shunted_s',
                'net': shunted_s(decode)}
    elif modelname == 'shunted_t':
        return {'modelname': 'shunted_t',
                'net': shunted_t(decode)}

    elif modelname == 'SegFormer_b1':
        return {'modelname': 'SegFormer_b1',
                'net': SegFormer(varients='b1')}
    elif modelname == 'SegFormer_b2':
        return {'modelname': 'SegFormer_b2',
                'net': SegFormer(varients='b2')}
    elif modelname == 'SegFormer_b3':
        return {'modelname': 'SegFormer_b3',
                'net': SegFormer(varients='b3')}
    elif modelname == 'SegFormer_b4':
        return {'modelname': 'SegFormer_b4',
                'net': SegFormer(varients='b4')}
    elif modelname == 'SegFormer_b5':
        return {'modelname': 'SegFormer_b5',
                'net': SegFormer(varients='b5')}

    elif modelname == 'swin_t':
        return {
            'modelname': 'swin_t',
            'net': swin(varients='swin_t')
        }
    elif modelname == 'swin_s':
        return {
            'modelname': 'swin_s',
            'net': swin(varients='swin_s')
        }
    elif modelname == 'swin_b':
        return {
            'modelname': 'swin_b',
            'net': swin(varients='swin_b')
        }
    elif modelname == 'pspnet':
        return {
            'modelname': 'pspnet',
            'net': PSPNet(num_classes=14)
        }

    elif modelname == 'deeplab':
        return {'modelname': 'Deeplab_v3+',
                'net': DeepLab(num_classes=14,backbone='resnet',output_stride=16,
                        sync_bn=None,freeze_bn=False)}

    elif modelname == 'unet':
        return {'modelname': 'UNet',
                'net': UNet(in_channels=3, num_classes=14, bilinear=True)}

    elif modelname == 'danet':
        return {'modelname': 'DANet',
                'net': DANet(num_classes=14)}

    elif modelname == 'fashion_s':
        return {
            'modelname': modelname + version,
            'net': Fashion_s()
        }
    elif modelname == 'fashion_t':
        return {
            'modelname': modelname + version,
            'net': Fashion_t()
        }
    elif modelname == 'fashion_b':
        return {
            'modelname': modelname + version,
            'net': Fashion_b()
        }
    elif modelname == 'semask_t':
        return {
            'modelname': modelname,
            'net': SeMask(variants='tiny')
        }
    elif modelname == 'semask_s':
        return {
            'modelname': modelname,
            'net': SeMask(variants='small')
        }
    elif modelname == 'semask_b':
        return {
            'modelname': modelname,
            'net': SeMask(variants='base')
        }

if __name__ == '__main__':
    import torch
    net = build_model(modelname='fashion_t')['net']
    x = torch.ones(4, 3, 224, 224)
    with torch.no_grad():
        y_dict = net(x)
        print(y_dict['output'].shape)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("#params: {:.2f}M".format(total_params/1e6))
    # flops, params = profile(net, (x,))
    # print('flops: ', flops, 'params: ', params)