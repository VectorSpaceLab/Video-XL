import math
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from .attention_temporal_videoae import *
from einops import rearrange, reduce, repeat

try:
    import xformers
    import xformers.ops as xops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

def silu(x):
    # swish
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch",'layer']
    if norm_type == "group":
        return torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)
    elif norm_type == "layer":
        return nn.LayerNorm(in_channels)
        
class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        tp=x.dtype
        x = x.float()

        # 执行填充操作
        x_padded = F.pad(x, self.pad_input, mode=self.padding_type)

        # 如果需要，将结果转换回 BFloat16
        x_padded = x_padded.to(tp)
        
        return self.conv(x_padded)

class TemporalAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        max_temporal_length=64,
    ):
        """
        a clean multi-head temporal attention
        """
        super().__init__()

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = Normalize(channels)
        self.qkv = zero_module(conv_nd(1, channels, channels * 3, 1))
        self.attention = QKVAttention(self.num_heads)
        self.relative_position_k = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.relative_position_v = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.proj_out = zero_module(
            conv_nd(1, channels, channels, 1)
        )  # conv_dim, in_channels, out_channels, kernel_size

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b h w) c t")
        # torch.Size([4608, 1152, 2])1
        # torch.Size([4608, 3456, 2])2
        # torch.Size([4608, 1152, 2])3
        # torch.Size([4608, 1152, 2])4
        #print(out.shape,end='1\n')
        qkv = self.qkv(self.norm(out))
        #print(qkv.shape,end='2\n')
        
        len_q = qkv.size()[-1]
        len_k, len_v = len_q, len_q

        k_rp = self.relative_position_k(len_q, len_k)
        v_rp = self.relative_position_v(len_q, len_v)  # [T,T,head_dim]
        out = self.attention(qkv, rp=(k_rp, v_rp))
        #print(out.shape,end='3\n')
        out = self.proj_out(out)
        #print(out.shape,end='4\n')
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)

        return x + out
class TemporalAttention_lin(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=8,
        num_head_channels=-1,
        max_temporal_length=64,
    ):
        """
        a clean multi-head temporal attention
        """
        super().__init__()

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        self.norm = nn.LayerNorm(channels)
        #self.norm = Normalize(channels)
        #self.qkv = zero_module(conv_nd(1, channels, channels * 3, 1))
        self.qkv = nn.Linear(channels, channels * 3)
        self.attention = QKVAttention(self.num_heads)
        self.relative_position_k = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.relative_position_v = RelativePosition(
            num_units=channels // self.num_heads,
            max_relative_position=max_temporal_length,
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b h w) t c")
        # torch.Size([4608, 1152, 2])1
        # torch.Size([4608, 3456, 2])2
        # torch.Size([4608, 1152, 2])3
        # torch.Size([4608, 1152, 2])4
        #print(out.shape,end='1\n')
        qkv = self.qkv(self.norm(out)).transpose(-1, -2)
        #print(qkv.shape,end='2\n')
        
        len_q = qkv.size()[-1]
        len_k, len_v = len_q, len_q

        k_rp = self.relative_position_k(len_q, len_k)
        v_rp = self.relative_position_v(len_q, len_v)  # [T,T,head_dim]
        
        out = self.attention(qkv, rp=(k_rp, v_rp))
        
        out = self.proj_out(out.transpose(-1, -2)).transpose(-1, -2)
        
        #print(out.shape,end='4\n')
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)

        return x + out
    
class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        # self.norm.to(x.device)
        # self.norm.to(x.dtype)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        # q = q.reshape(b,c,h*w) # bcl
        # q = q.permute(0,2,1)   # bcl -> blc l=hw
        # k = k.reshape(b,c,h*w) # bcl
        q = rearrange(q, "b c t h w -> (b t) (h w) c")  # blc
        k = rearrange(k, "b c t h w -> (b t) c (h w)")  # bcl

        w_ = torch.bmm(q, k)  # b,l,l
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # v = v.reshape(b,c,h*w)
        v = rearrange(v, "b c t h w -> (b t) c (h w)")  # bcl

        # attend to values
        w_ = w_.permute(0, 2, 1)  # bll
        h_ = torch.bmm(v, w_)  # bcl

        # h_ = h_.reshape(b,c,h,w)
        h_ = rearrange(h_, "(b t) c (h w) -> b c t h w", b=b, h=h)

        h_ = self.proj_out(h_)

        return x + h_
    
class MultiHeadAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        self.norm = nn.LayerNorm(in_channels)
        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, t, h, w = x.shape
        #print(x.shape)
        # Normalize and reshape input
        h_ = rearrange(x, "b c t h w -> (b t) (h w) c")
        h_ = self.norm(h_)

        # Linear projections
        q = self.q_linear(h_)
        k = self.k_linear(h_)
        v = self.v_linear(h_)

        # Reshape to multi-head
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h l d -> b l (h d)")

        # Project back to original dimension
        out = self.proj_out(out)

        # Reshape back to original shape
        out = rearrange(out, "(b t) (h w) c -> b c t h w", b=b, h=h, t=t)
        #print(out.shape)
        return x + out