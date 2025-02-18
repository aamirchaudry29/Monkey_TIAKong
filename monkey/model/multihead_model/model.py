from collections import OrderedDict

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.initialization as init
import timm

# from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from torchvision.ops import Conv2dNormActivation


def load_encoder_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder_weights = {
        k: v
        for k, v in checkpoint["model_state_dict"].items()
        if "encoder" in k
    }
    model.load_state_dict(encoder_weights, strict=False)
    return model


class TimmEncoderFixed(nn.Module):
    def __init__(
        self,
        name,
        pretrained=True,
        in_channels=3,
        depth=5,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if drop_path_rate is None:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
            )
        else:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


def get_timm_encoder(
    name,
    in_channels=3,
    depth=5,
    weights=False,
    output_stride=32,
    drop_rate=0.5,
    drop_path_rate=0.25,
):
    encoder = TimmEncoderFixed(
        name,
        weights,
        in_channels,
        depth,
        output_stride,
        drop_rate,
        drop_path_rate,
    )
    return encoder


def get_multihead_model(
    enc="convnextv2_large.fcmae_ft_in22k_in1k",
    pretrained=True,
    num_heads=3,
    decoders_out_channels=[1, 1, 1],
    use_batchnorm=False,
    attention_type=None,
    center=False,
):
    pre_path = None
    if type(pretrained) == str:
        pre_path = pretrained
        pretrained = False
    
    # deal with large pooling in convnext type models:
    next = False
    if "next" in enc:
        depth = 4
        next = True
    else: 
        depth = 5

    if "efficientvit" in enc:
        depth = 4
        next = True
        encoder = get_timm_encoder(
            name=enc,
            in_channels=3,
            depth=depth,
            weights=pretrained,
            output_stride=32,
            drop_path_rate=None
        )
    else:
        encoder = get_timm_encoder(
            name=enc,
            in_channels=3,
            depth=depth,
            weights=pretrained,
            output_stride=32,
            drop_rate=0.5,
            drop_path_rate=0.25,
        )

    decoder_channels = (512, 256, 128, 64, 32)[:depth]

    decoders = []
    for i in range(num_heads):
        decoders.append(
            UnetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                use_batchnorm=use_batchnorm,
                center=center,
                attention_type=attention_type,
                next=next,
            )
        )

    heads = []
    for i in range(num_heads):
        heads.append(
            smp.base.SegmentationHead(
                in_channels=decoders[i]
                .blocks[-1]
                .conv2[0]
                .out_channels,
                out_channels=decoders_out_channels[
                    i
                ],  # instance channels
                activation=None,
                kernel_size=1,
            )
        )

    model = MultiHeadModel(encoder, decoders, heads)
    if pre_path:
        state_dict = torch.load(pre_path, map_location=f"cpu")[
            "model_state_dict"
        ]
        new_state = model.state_dict()
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_state[k] = v
        model.load_state_dict(new_state)
    return model


class SubPixelUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(SubPixelUpsample, self).__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels * upscale_factor ** 2,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.up = SubPixelUpsample(in_channels, in_channels, upscale_factor=2)
        self.conv1 = Conv2dNormActivation(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention2 = md.Attention(
            attention_type, in_channels=out_channels
        )


    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.attention = md.Attention("scse", in_channels=in_channels)
    
    def forward(self, x):
        x = self.attention(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=False,
        attention_type=None,
        center=False,
        next=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels,
                head_channels,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_batchnorm=use_batchnorm, attention_type=attention_type
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels
            )
        ]
        if next:
            blocks.append(
                DecoderBlock(
                    out_channels[-1],
                    0,
                    out_channels[-1] // 2,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[
            1:
        ]  # remove first skip with same spatial resolution
        features = features[
            ::-1
        ]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class MultiHeadModel(torch.nn.Module):
    def __init__(self, encoder, decoder_list, head_list):
        super(MultiHeadModel, self).__init__()
        self.encoder = nn.ModuleList([encoder])[0]
        self.decoders = nn.ModuleList(decoder_list)
        self.heads = nn.ModuleList(head_list)
        self.initialize()

    def initialize(self):
        for decoder in self.decoders:
            init.initialize_decoder(decoder)
        for head in self.heads:
            init.initialize_head(head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features))

        masks = []
        for head, decoder_output in zip(self.heads, decoder_outputs):
            masks.append(head(decoder_output))

        return torch.cat(masks, 1)


def freeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model


def unfreeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = True
    return model
