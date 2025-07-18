from torch import nn
import timm
import torch
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from mlru_pp.network_architecture.layers import LayerNorm
from mlru_pp.network_architecture.acdc.transformerblock import TransformerBlock
from mlru_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
#from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5

einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],
                                                     proj_size=proj_size[i], num_heads=num_heads,
                                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
class UnetrPPEncoder1(nn.Module):
    def __init__(
        self,
        input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],
        dims=[64, 128, 320, 512],
        proj_size=[64, 64, 64, 32],
        depths=[3, 4, 6, 3],
        num_heads=4,
        spatial_dims=3,
        in_channels=1,
        dropout=0.0,
        transformer_dropout_rate=0.1,
        embed_dim=256,
        seq_len_out=50,
        **kwargs
    ):
        super().__init__()
        self.pvt = pvt_v2_b2(in_chans=in_channels)  # pass in_chans to support 1-channel input
        self.embed_dim = embed_dim
        self.seq_len_out = seq_len_out

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.seq_proj = nn.Linear(16, seq_len_out)  # Assumes D=16 during projection

    def forward_features(self, x):
        B, C, D, H, W = x.shape
        x_2d = x.view(B * D, C, H, W)

        with torch.no_grad():
            feat_list = self.pvt.forward_features(x_2d)  # list of 4 stages

        hidden_states = []
        for f in feat_list[:-1]:
            try:
                f = f.view(B, D, f.shape[1], f.shape[2], f.shape[3])  # (B, D, C, H, W)
                f = f.permute(0, 2, 1, 3, 4).contiguous()              # (B, C, D, H, W)
                hidden_states.append(f)
            except Exception as e:
                print(f"Skip stage due to shape mismatch: {f.shape}, Error: {e}")

        final_feat = feat_list[-1]  # (B*D, 512, h, w)
        f = final_feat.flatten(2).mean(dim=-1)  # (B*D, 512)
        f = f.view(B, D, -1)                   # (B, D, 512)

        gru_out, _ = self.gru(f)               # (B, D, 256)
        x_proj = gru_out.transpose(1, 2)       # (B, 256, D)
        x_proj = self.seq_proj(x_proj)         # (B, 256, 50)
        x_proj = x_proj.transpose(1, 2)        # (B, 50, 256)

        return x_proj, hidden_states

    def forward(self, x):
        x_proj, hidden_states = self.forward_features(x)
        return x_proj, hidden_states
class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out
