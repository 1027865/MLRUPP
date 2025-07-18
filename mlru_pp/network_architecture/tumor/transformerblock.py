import torch.nn as nn
import torch
from mlru_pp.network_architecture.dynunet_block import UnetResBlock
import math

from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# class EPA(nn.Module):
#     """
#         Efficient Paired Attention Block, based on: "Shaker et al.,
#         UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
#         """
#     def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
#                  channel_attn_drop=0.1, spatial_attn_drop=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

#         # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
#         self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

#         # E and F are projection matrices with shared weights used in spatial attention module to project
#         # keys and values from HWD-dimension to P-dimension
#         self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

#         self.attn_drop = nn.Dropout(channel_attn_drop)
#         self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

#     def forward(self, x):
#         B, N, C = x.shape

#         qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
#         qkvv = qkvv.permute(2, 0, 3, 1, 4)
#         q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

#         q_shared = q_shared.transpose(-2, -1)
#         k_shared = k_shared.transpose(-2, -1)
#         v_CA = v_CA.transpose(-2, -1)
#         v_SA = v_SA.transpose(-2, -1)

#         proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
#         k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF)))

#         q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
#         k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

#         attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
#         attn_CA = attn_CA.softmax(dim=-1)
#         attn_CA = self.attn_drop(attn_CA)
#         x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

#         attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
#         attn_SA = attn_SA.softmax(dim=-1)
#         attn_SA = self.attn_drop_2(attn_SA)
#         x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

#         return x_CA + x_SA

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'temperature', 'temperature2'}
import math
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
class LightCBAM(nn.Module):
    """
    Lightweight CBAM-style replacement for EPA.
    Preserves signature: [B, N, C] -> [B, N, C]
    """
    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        reduced = max(4, hidden_size // 16)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(hidden_size, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.ca_drop = nn.Dropout(channel_attn_drop)

        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sa_drop = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, N, C = x.shape
       

        # Infer H, W from N
        H = int(math.sqrt(N))
        W = math.ceil(N / H)
        pad_len = H * W - N
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad N dimension

        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]

        # Channel Attention
        ca = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        x = self.ca_drop(x * ca)

        # Spatial Attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.sa_conv(torch.cat([avg, max_], dim=1)))
        x = self.sa_drop(x * sa)

        # Back to [B, N, C]
        x = x.view(B, C, H * W).permute(0, 2, 1)
        if pad_len > 0:
            x = x[:, :-pad_len, :]

       
        return x
class EPA(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, mscb_kernel_sizes=[1, 3, 5],
                 num_heads=4, qkv_bias=False, channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.epa = LightCBAM(input_size, hidden_size, proj_size, num_heads,
                                 qkv_bias, channel_attn_drop, spatial_attn_drop)

        self.hidden_size = hidden_size
        self.reshape_required = True  # controls whether we go to [B, C, H, W]

        self.mscb = MSCB(
            in_channels=hidden_size,
            out_channels=hidden_size,
            stride=1,
            kernel_sizes=mscb_kernel_sizes,
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )

    def forward(self, x):
        # x: [B, N, C]
        x = self.epa(x)  # still [B, N, C]
        

        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = math.ceil(N / H)
        pad_len = H * W - N

        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad sequence dim

        x_reshaped = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
        x_mscb = self.mscb(x_reshaped)  # [B, C, H, W]

        x_out = x_mscb.view(B, C, H * W).permute(0, 2, 1)  # [B, N_pad, C]
        if pad_len > 0:
            x_out = x_out[:, :-pad_len, :]

       
        return x_out
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}
