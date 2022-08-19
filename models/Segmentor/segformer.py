from models.backbone.mix_transformer import mit_b5, mit_b4, mit_b3, mit_b2, mit_b1, mit_b0
from models.decode_head.segformer_head import SegFormerHead
import torch.nn.functional as F
import torch.nn as nn


class SegFormer(nn.Module):
    def __init__(self, varients='b2', num_classes=14):
        super(SegFormer, self).__init__()
        if varients == 'b0':
            self.backbone = mit_b0()
        elif varients == 'b1':
            self.backbone = mit_b1()
        elif varients == 'b2':
            self.backbone = mit_b2()
        elif varients == 'b3':
            self.backbone = mit_b3()
        elif varients == 'b4':
            self.backbone = mit_b4()
        elif varients == 'b5':
            self.backbone = mit_b5()

        self.decode_head = SegFormerHead(in_channels=self.backbone.embed_dims,
                                         embedding_dim=768,
                                         num_classes=num_classes)

    def forward(self, x):
        out_size = x.size()[2:]
        feats = self.backbone(x)
        output = self.decode_head(feats)
        output = F.interpolate(output, size=out_size, mode='bilinear', align_corners=False)

        return {'output':output}


if __name__ == "__main__":
    """
    b0: 3.7M
    b1: 13.7M
    b2: 27.5M
    b3: 47.3M
    b4: 64.1M
    b5: 84.7M
    """
    import torch
    model = SegFormer(varients='b3', num_classes=14)
    inputs = torch.rand(4, 3, 224, 224)
    outputs = model(inputs)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Params: {:.2f}".format(params / 1e6))
    print(outputs['output'].size())
