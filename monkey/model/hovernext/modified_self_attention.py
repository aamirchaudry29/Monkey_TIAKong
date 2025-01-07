from collections import OrderedDict

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.initialization as init
import timm

# from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md


def load_encoder_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    encoder_weights = {
        k: v
        for k, v in checkpoint["model_state_dict"].items()
        if "encoder" in k
    }
    model.load_state_dict(encoder_weights, strict=False)
    return model


def load_checkpoint(model, cp_path, rank=0):
    cp = torch.load(cp_path, map_location=f"cuda:{rank}")
    step = cp["step"]
    try:
        best_loss = cp["best_loss"]
    except KeyError:
        # CAREFUL, if the metric is to be minized, this should be set to inf
        best_loss = 0
    try:
        model.load_state_dict(cp["model_state_dict"])

        print("succesfully loaded checkpoint step", step)
    except:
        print("trying secondary checkpoint loading")
        state_dict = cp["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[
                7:
            ]  # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("succesfully loaded checkpoint step", step)
    return model, step, best_loss


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


def get_model(
    enc="convnextv2_large.fcmae_ft_in22k_in1k",
    pretrained=True,
    num_heads=3,
    decoders_out_channels=[1, 1, 1],
    use_batchnorm=False,
    attention_type=None,
):
    pre_path = None
    if type(pretrained) == str:
        pre_path = pretrained
        pretrained = False
    # small fix to deal with large pooling in convnext type models:
    # depth = 4 if "next" in enc else 5
    depth = 3

    encoder = get_timm_encoder(
        name=enc,
        in_channels=3,
        depth=depth,
        weights=pretrained,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.5,
    )
    decoder_channels = (256, 128, 64, 32, 16)[:depth]

    decoders = []
    for i in range(num_heads):
        decoders.append(
            UnetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                use_batchnorm=use_batchnorm,
                center=False,
                attention_type=attention_type,
                next="next" in enc,
            )
        )

    # heads = []
    # for i in range(num_heads):
    #     heads.append(
    #         smp.base.SegmentationHead(
    #             in_channels=decoders[i]
    #             .blocks[-1]
    #             .conv2[0]
    #             .out_channels,
    #             out_channels=decoders_out_channels[
    #                 i
    #             ],  # instance channels
    #             activation=None,
    #             kernel_size=1,
    #         )
    #     )

    model = MultiHeadModel(encoder, decoders)
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


class ModifiedSelfAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=1, embed_dim=1, kernel_size=3, num_heads=1, patch_size=16, img_size=256): 
        super(ModifiedSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.number_of_patches = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.number_of_patches, embed_dim))
        
        # Query, key, value projections
        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        self.out_projection = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, feature_map1, feature_map2):
        """
        feature_map1: (batch_size, in_channels, height, width)
        feature_map2: (batch_size, in_channels, height, width)
        """
        B, C, H, W = feature_map1.size()

        # Compute Query, Key, Value for feature_map1
        proj_query = self.query_conv(x).view(
            batch, self.out_channels, height * width
        )
        proj_key = self.key_conv(x).view(
            batch, self.out_channels, height * width
        )
        proj_value = self.value_conv(x).view(
            batch, self.out_channels, height * width
        )

        # Calculating attention scores
        energy = (proj_query * proj_key).sum(
            dim=2
        )  # (batch_size, out_channels)
        print(energy.size())
        attention = F.softmax(energy, dim=1)

        # Getting the weighted sum of values
        # (batch_size, height*width, out_channels)
        print(proj_value.size(), attention.size())
        out = attention * proj_value
        out = out.permute(0, 2, 1).view(
            batch, self.out_channels, height, width
        )

        out = out + x
        return out


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU()

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


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
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(
            attention_type, in_channels=out_channels
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


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
    def __init__(self, encoder, decoder_list):
        super(MultiHeadModel, self).__init__()
        self.encoder = nn.ModuleList([encoder])[0]
        self.decoders = nn.ModuleList(decoder_list)
        self.initialize()

    def initialize(self):
        for decoder in self.decoders:
            init.initialize_decoder(decoder)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features))

        return torch.cat(decoder_outputs, 1)


def freeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    return model


def unfreeze_enc(model):
    for p in model.encoder.parameters():
        p.requires_grad = True
    return model
