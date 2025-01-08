from collections import OrderedDict

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.initialization as init
import timm

# from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from monkey.model.hovernext.model import get_timm_encoder, UnetDecoder


def get_modified_hovernext(
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
    depth = 4 if "next" in enc else 5
    # depth = 3

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

    heads = []
    for i in range(num_heads):
        heads.append(
            smp.base.SegmentationHead(
                in_channels=48,
                out_channels=decoders_out_channels[
                    i
                ],  # instance channels
                activation=None,
                kernel_size=1,
            )
        )

    model = Modified_MultiHeadModel(encoder, decoders, heads)
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
    


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_pool).view(b, c, 1, 1)
        max_pool = self.max_pool(x).view(b, c)
        max_out = self.fc(max_pool).view(b, c, 1, 1)
        y = self.sigmoid(avg_out + max_out)
        return x * y
    

class PatchMultiheadAttention(nn.Module):
    def __init__(self, in_channels=48, patch_size=16, embed_dim=256, num_heads=8):
        super(PatchMultiheadAttention, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Linear layer to embed flattened patches
        self.projection = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

        # Multihead Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Final linear layer to project back
        self.fc = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, in_channels, height, width = x.size()
        
        # Ensure dimensions are divisible by patch size
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Image size must be divisible by patch size"

        # Reshape into patches
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, in_channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, -1, in_channels * self.patch_size * self.patch_size)
        
        # Linear projection of patches
        patch_embeddings = self.projection(patches)  # Shape: (batch_size, num_patches, embed_dim)

        # Apply multihead attention
        attn_output, _ = self.multihead_attn(patch_embeddings, patch_embeddings, patch_embeddings)  # Shape: (batch_size, num_patches, embed_dim)

        # Project back to original patch shape
        attn_output = self.fc(attn_output)  # Shape: (batch_size, num_patches, in_channels * patch_size * patch_size)
        attn_output = attn_output.view(batch_size, num_patches_h * num_patches_w, in_channels, self.patch_size, self.patch_size)
        attn_output = attn_output.permute(0, 2, 1, 3, 4).contiguous().view(batch_size, in_channels, height, width)

        return attn_output



class Modified_MultiHeadModel(torch.nn.Module):
    def __init__(self, encoder, decoder_list, head_list):
        super(Modified_MultiHeadModel, self).__init__()
        self.encoder = nn.ModuleList([encoder])[0]
        self.decoders = nn.ModuleList(decoder_list)
        self.heads = nn.ModuleList(head_list)
        self.CAM_Modules = nn.ModuleList(
            [ChannelAttention(in_channels=48) for i in range(3)]
        )
        # self.CAM_Modules = nn.ModuleList(
        #     [PatchMultiheadAttention(in_channels=48) for i in range(3)]
        # )
        self.initialize()

    def initialize(self):
        for decoder in self.decoders:
            init.initialize_decoder(decoder)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features)) # 48 channels
        decoder_outputs = torch.cat(decoder_outputs, 1)

        head_outputs = []
        for i, cam in enumerate(self.CAM_Modules):
            weighted_feature_map = cam(decoder_outputs)
            head_output = self.heads[i](weighted_feature_map)
            head_outputs.append(head_output)

        return torch.cat(head_outputs, 1)