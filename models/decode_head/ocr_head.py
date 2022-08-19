# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.
    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.
    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats) # Conv projections
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        # query: B C H W -> B (H W) C
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)       # Conv projections
        value = self.value_project(key_feats)   # Conv projections

        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(*key.shape[:2], -1)       # key: B C H W -> B C (H W)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() # value: B C H W -> B (H W) C

        sim_map = torch.matmul(query, key)  # similarity map: query * key
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map    # qk/sqrt(d)
        sim_map = F.softmax(sim_map, dim=-1)            # softmax(qk/sqrt(d))

        context = torch.matmul(sim_map, value)       # softmax(qk/sqrt(d))v
        context = context.permute(0, 2, 1).contiguous() # B N C -> B C N
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])   # B C N -> B C H W
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class SpatialGatherModule(nn.Module):
    """
    Aggregate the context features according to the initial predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1) # B K (H W)
        feats = feats.view(batch_size, channels, -1)    # B C (H W)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)  # B (H W) C
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)    # B K (H W)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)    # B K C
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)    # B C K d
        return ocr_context


class ObjectAttentionBlock(SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg, act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock, self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        # if self.query_downsample is not None:
        #     output = resize(query_feats)
        return output


class OCRHead(nn.Module):
    """Object-Contextual Representations for Semantic Segmentation.
    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.
    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """
    def __init__(self, num_class=14, ocr_channels=256, scale=1, in_channels=2048, channels=512,):
        super(OCRHead, self).__init__()
        self.ocr_channels = ocr_channels
        self.in_channels = in_channels
        self.channels = channels
        self.scale = scale

        self.conv_cfg = dict(type='Conv2d')
        self.norm_cfg = dict(type='BN')
        self.act_cfg = dict(type='ReLU')

        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(self.channels, num_class, kernel_size=1, stride=1)
        )

    def forward(self, x, prev_output):
        """
        Forward function.
        Args:
            inputs: inputs from multi scale stages
            prev_output: output from last decode head
        """
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        # query: feats; key: context
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)
        return output


if __name__ == '__main__':
    x = torch.randn(4, 2048, 28, 28)
    prev = torch.randn(4, 14, 28, 28)
    module = OCRHead(num_class=14)
    output = module(x, prev)
    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print("{}".format(params/1e6))
    print(output.shape)

