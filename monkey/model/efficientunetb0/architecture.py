""" Full assembly of the parts to form the complete network """

import torch
from torchvision.models.efficientnet import (
    EfficientNet,
    EfficientNet_B0_Weights,
    MBConv,
    MBConvConfig,
    efficientnet_b0,
)
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

from monkey.model.efficientunetb0.layers import *

##################################################
##################################################
# Custom implementation of the PyTorch Version of Efficient U-Net, proposed from paper "Robust interactive semantic segmentation of pathology images with minimal user input"
# Author: Kesi Xu
##################################################
##################################################


MBConv1_conf = MBConvConfig(
    expand_ratio=1,
    kernel=3,
    stride=1,
    input_channels=1280,
    out_channels=320,
    num_layers=1,
)
MBConv2_conf = MBConvConfig(
    expand_ratio=6,
    kernel=5,
    stride=1,
    input_channels=432,
    out_channels=192,
    num_layers=4,
)  # upsample   deblock2_concatenation with block5c_add output is (112,14,14)
MBConv3_conf = MBConvConfig(
    expand_ratio=6,
    kernel=5,
    stride=1,
    input_channels=192,
    out_channels=112,
    num_layers=3,
)  #
MBConv4_conf = MBConvConfig(
    expand_ratio=6,
    kernel=3,
    stride=1,
    input_channels=152,
    out_channels=80,
    num_layers=3,
)  # upsample    skip_conn='block3b_add'
MBConv5_conf = MBConvConfig(
    expand_ratio=6,
    kernel=5,
    stride=1,
    input_channels=104,
    out_channels=40,
    num_layers=2,
)  # upsample    skip_conn='block2b_add'
MBConv6_conf = MBConvConfig(
    expand_ratio=6,
    kernel=3,
    stride=1,
    input_channels=56,
    out_channels=24,
    num_layers=2,
)  # upsample    skip_conn='block1a_project_bn'
MBConv7_conf = MBConvConfig(
    expand_ratio=1,
    kernel=3,
    stride=1,
    input_channels=24,
    out_channels=16,
    num_layers=2,
)


__all__ = ["EfficientUnet_MBConv", "get_efficientunet_b0_MBConv"]
# __all__ = ['EfficientUnet_MBConv', 'get_efficientunet_b0_MBConv', 'get_efficientunet_b1_MBConv', 'get_efficientunet_b2_MBConv',
#            'get_efficientunet_b3_MBConv', 'get_efficientunet_b4_MBConv', 'get_efficientunet_b5_MBConv', 'get_efficientunet_b6_MBConv',
#            'get_efficientunet_b7_MBConv']


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class EfficientUnet_MBConv(nn.Module):
    def __init__(self, encoder, out_channels=1):
        super().__init__()
        self.net_name = "efficientunet_mbconv"
        self.encoder = encoder
        self.n_classes = 1  # default setting for evaluation in the training process
        # self.skip_connections = create_feature_extractor(self.encoder,
        #                                                  return_nodes={"features.1.0.block.2": "layer0", "features.2.1.add": "layer1",
        #                                                                "features.3.1.add": "layer2", "features.5.1.add": "layer3","features.8":'encoder_output'})
        self.skip_connections = create_feature_extractor(
            self.encoder,
            return_nodes={
                "1.0.block.2": "layer0",
                "2.1.add": "layer1",
                "3.1.add": "layer2",
                "5.1.add": "layer3",
                "8": "encoder_output",
            },
        )

        self.block2_upsample = nn.ConvTranspose2d(
            320, 320, kernel_size=2, stride=2
        )  # implement transpose2DConv, increase the image size twice.from (7,7,320) to (14,14,320)
        self.block4_upsample = nn.ConvTranspose2d(
            112, 112, kernel_size=2, stride=2
        )  # from (14, 14, 112) to (28, 28, 112)
        self.block5_upsample = nn.ConvTranspose2d(
            80, 80, kernel_size=2, stride=2
        )  # from (28, 28, 80) to (56, 56, 80)
        self.block6_upsample = nn.ConvTranspose2d(
            40, 40, kernel_size=2, stride=2
        )  # from (56, 56, 40) to (112, 112, 40)
        self.block7_upsample = nn.ConvTranspose2d(
            24, 24, kernel_size=2, stride=2
        )  # #from (112, 112, 24) to (224,224,24)
        self.MBConv1 = MBConv(
            MBConv1_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )
        self.MBConv2 = MBConv(
            MBConv2_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )  # upsample   deblock2_concatenation with block5c_add output is (112,14,14)
        self.MBConv3 = MBConv(
            MBConv3_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )  #
        self.MBConv4 = MBConv(
            MBConv4_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )  # upsample    skip_conn='block3b_add'
        self.MBConv5 = MBConv(
            MBConv5_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )  # upsample    skip_conn='block2b_add'
        self.MBConv6 = MBConv(
            MBConv6_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )  # upsample    skip_conn='block1a_project_bn'
        self.MBConv7 = MBConv(
            MBConv7_conf,
            stochastic_depth_prob=0.2,
            norm_layer=nn.BatchNorm2d,
        )

        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        input_ = x  # RGB 3 channels for the optional input concate with the final layer

        # encoder_output = self.encoder.features(input)
        encoder_output = self.encoder(x)
        skip_connections_out = self.skip_connections(x)

        # block1
        x = self.MBConv1(
            encoder_output
        )  # input_filters=1280, output_filters=320,num_repeat=1

        # block2
        x = self.block2_upsample(x)  # from (7,7,320) to (14,14,320)

        x = torch.cat(
            [x, skip_connections_out["layer3"]], dim=1
        )  # input=320, output=320+112=432  concate with block5c_add[0][0](which is MBConvBlock-120)
        x = self.MBConv2(
            x
        )  # num_repeat=4, input_filters=432, output_filters=192
        #
        # block 3
        x = self.MBConv3(
            x
        )  # num_repeat=3, input_filters=192, output_filters=112

        # block4
        x = self.block4_upsample(x)  # from (14,14,112) to (28,28,112)
        x = torch.cat(
            [x, skip_connections_out["layer2"]], dim=1
        )  # output= 112 + 40 = 152  concate with block3b_add(which is MBConvBlock-60)
        x = self.MBConv4(
            x
        )  # num_repeat=3 , input= 112(152) output = 80

        # block 5
        x = self.block5_upsample(x)
        x = torch.cat(
            [x, skip_connections_out["layer1"]], dim=1
        )  # output= 80 + 24 = 104  concate with block2b_add(which is MBConvBlock-36)
        x = self.MBConv5(x)  # num_repeat=2 , input 80,  output=40

        # block 6
        x = self.block6_upsample(x)
        x = torch.cat(
            [x, skip_connections_out["layer0"]], dim=1
        )  # output= 40 + 16 = 56  concate with block1a_project_bn(which is MBConvBlock-12)
        x = self.MBConv6(x)  # num_repeat=2 , input=40,  output=24

        # block 7
        x = self.block7_upsample(
            x
        )  # from (112, 112, 24) to (224,224,24)
        x = self.MBConv7(x)  # num_repeat=2 , input=24,  output=16

        x = self.final_conv(
            x
        )  # input 32 channels, output is 1 channel BW image
        #
        return x


def get_efficientunet_b0_MBConv(
    out_channels=1, concat_input=True, pretrained=True
):
    if pretrained:
        encoder = efficientnet_b0(
            weights="EfficientNet_B0_Weights.DEFAULT"
        )
    else:
        encoder = efficientnet_b0()

    encoder = encoder.features
    model = EfficientUnet_MBConv(encoder, out_channels=out_channels)
    #
    return model
