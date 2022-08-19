import torch.nn as nn
import torch.nn.functional as F
from models.backbone.semask_swin import SeMaskSwinTransformer
from models.decode_head.branch_fpn_head import SemanticBranchFPN
from models.decode_head.upernet import UPerHead


class SeMask(nn.Module):
    def __init__(self, variants='tiny', num_classes=14):
        super(SeMask, self).__init__()
        if variants == 'tiny':
            dim = 96
            self.backbone = SeMaskSwinTransformer(
                embed_dim=dim,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                num_cls=num_classes,
                sem_window_size=7,
                num_sem_blocks=[1, 1, 1, 1],
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
            )

        elif variants == 'small':
            dim = 96
            self.backbone = SeMaskSwinTransformer(
                embed_dim=dim,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                num_cls=num_classes,
                sem_window_size=7,
                num_sem_blocks=[1, 1, 1, 1],
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
            )

        elif variants == 'base':
            dim = 128
            self.backbone = SeMaskSwinTransformer(
                embed_dim=dim,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                num_cls=num_classes,
                sem_window_size=12,
                num_sem_blocks=[1, 1, 1, 1],
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
            )
        else:
            raise NotImplementedError

        feature_strides = (4, 8, 16, 32)
        in_channels = [dim * (2 ** i) for i in range(4)]

        self.head = UPerHead(in_channels=in_channels, channels=256, num_classes=num_classes)
        self.semantic_head = SemanticBranchFPN(
            feature_strides=feature_strides,
            num_classes=num_classes)

    def forward(self, inputs):
        oue_size = inputs.shape[2:]
        x, cls_x = self.backbone(inputs)
        output = self.head(x)
        cls_output = self.semantic_head(cls_x)
        output = F.interpolate(output, size=oue_size, mode='bilinear', align_corners=False)
        cls_output = F.interpolate(cls_output, size=oue_size, mode='bilinear', align_corners=False)
        return {'output': output, 'aux': cls_output}


if __name__ == '__main__':
    import torch
    imgs = torch.ones(4, 3, 224, 224)
    net = SeMask(variants='small', num_classes=14)
    out_dict = net(imgs)
    print(out_dict['output'].shape)
    print(out_dict['aux'].shape)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("#params: {:.2f}M".format(total_params / 1e6))
