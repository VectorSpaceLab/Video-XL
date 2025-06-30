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
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
        padding_type="replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type
        )
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h
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

        qkv = self.qkv(self.norm(out))

        len_q = qkv.size()[-1]
        len_k, len_v = len_q, len_q

        k_rp = self.relative_position_k(len_q, len_k)
        v_rp = self.relative_position_v(len_q, len_v)  # [T,T,head_dim]
        out = self.attention(qkv, rp=(k_rp, v_rp))

        out = self.proj_out(out)
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)

        return x + out
    
    
class RESMLP(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.activation_fn = ACT2FN['gelu']
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h_=hidden_states
        hidden_states=hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states=hidden_states.permute(0, 4, 1, 2, 3)
        
        return hidden_states+h_
    
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
class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        patch_size=1,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.patch_size = patch_size
        patch_dim = query_dim * patch_size * patch_size
        self.norm = nn.LayerNorm(patch_dim)

        self.to_q = nn.Linear(patch_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, patch_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        b, c, t, height, width = x.shape

        # patch: [patch_size, patch_size]
        divide_factor_height = height // self.patch_size
        divide_factor_width = width // self.patch_size
        x = rearrange(
            x,
            "b c t (df1 ph) (df2 pw) -> (b t) (df1 df2) (ph pw c)",
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.norm(x)

        context = default(context, x)
        context = repeat(context, "b n d -> (b t) n d", b=b, t=t)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), (q, k, v)
        )

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b t h) () j", t=t, h=self.heads)

        if XFORMERS_IS_AVAILBLE:
            if exists(mask):
                mask = mask.to(q.dtype)
                max_neg_value = -torch.finfo(q.dtype).max

                attn_bias = torch.zeros_like(mask)
                attn_bias.masked_fill_(mask <= 0.5, max_neg_value)

                mask = mask.detach().cpu()
                attn_bias = attn_bias.expand(-1, q.shape[1], -1)

                attn_bias_expansion_q = (attn_bias.shape[1] + 7) // 8 * 8
                attn_bias_expansion_k = (attn_bias.shape[2] + 7) // 8 * 8

                attn_bias_expansion = torch.zeros(
                    (attn_bias.shape[0], attn_bias_expansion_q, attn_bias_expansion_k),
                    dtype=attn_bias.dtype,
                    device=attn_bias.device,
                )
                attn_bias_expansion[:, : attn_bias.shape[1], : attn_bias.shape[2]] = (
                    attn_bias
                )

                attn_bias = attn_bias.detach().cpu()

                out = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=attn_bias_expansion[
                        :, : attn_bias.shape[1], : attn_bias.shape[2]
                    ],
                    scale=self.scale,
                )
            else:
                out = xops.memory_efficient_attention(q, k, v, scale=self.scale)
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~(mask > 0.5), max_neg_value)
            attn = sim.softmax(dim=-1)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        ret = self.to_out(out)
        ret = rearrange(
            ret,
            "(b t) (df1 df2) (ph pw c) -> b c t (df1 ph) (df2 pw)",
            b=b,
            t=t,
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return ret
class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        patch_size=1,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        # print(f"query dimension is {query_dim}")

        self.patch_size = patch_size
        patch_dim = query_dim * patch_size * patch_size
        self.norm = nn.LayerNorm(patch_dim)

        self.to_q = nn.Linear(patch_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, patch_dim), nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        b, c, t, height, width = x.shape

        # patch: [patch_size, patch_size]
        divide_factor_height = height // self.patch_size
        divide_factor_width = width // self.patch_size
        x = rearrange(
            x,
            "b c t (df1 ph) (df2 pw) -> (b t) (df1 df2) (ph pw c)",
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.norm(x)

        context = default(context, x)
        context = repeat(context, "b n d -> (b t) n d", b=b, t=t)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads), (q, k, v)
        )

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b t h) () j", t=t, h=self.heads)

        if XFORMERS_IS_AVAILBLE:
            if exists(mask):
                mask = mask.to(q.dtype)
                max_neg_value = -torch.finfo(q.dtype).max

                attn_bias = torch.zeros_like(mask)
                attn_bias.masked_fill_(mask <= 0.5, max_neg_value)

                mask = mask.detach().cpu()
                attn_bias = attn_bias.expand(-1, q.shape[1], -1)

                attn_bias_expansion_q = (attn_bias.shape[1] + 7) // 8 * 8
                attn_bias_expansion_k = (attn_bias.shape[2] + 7) // 8 * 8

                attn_bias_expansion = torch.zeros(
                    (attn_bias.shape[0], attn_bias_expansion_q, attn_bias_expansion_k),
                    dtype=attn_bias.dtype,
                    device=attn_bias.device,
                )
                attn_bias_expansion[:, : attn_bias.shape[1], : attn_bias.shape[2]] = (
                    attn_bias
                )

                attn_bias = attn_bias.detach().cpu()

                out = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=attn_bias_expansion[
                        :, : attn_bias.shape[1], : attn_bias.shape[2]
                    ],
                    scale=self.scale,
                )
            else:
                out = xops.memory_efficient_attention(q, k, v, scale=self.scale)
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~(mask > 0.5), max_neg_value)
            attn = sim.softmax(dim=-1)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        ret = self.to_out(out)
        ret = rearrange(
            ret,
            "(b t) (df1 df2) (ph pw c) -> b c t (df1 ph) (df2 pw)",
            b=b,
            t=t,
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return ret