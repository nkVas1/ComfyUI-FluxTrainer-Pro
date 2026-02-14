# copy from FLUX repo: https://github.com/black-forest-labs/flux
# license: Apache-2.0 License

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Union

from .device_utils import init_ipex
from .custom_offloading_utils import ModelOffloader
init_ipex()

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

# USE_REENTRANT = True


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    mlp_gated: bool = False


# region autoencoder


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


# endregion
# region config


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    # repo_id: str | None
    # repo_flow: str | None
    # repo_ae: str | None


configs = {
    "dev": ModelSpec(
        # repo_id="black-forest-labs/FLUX.1-dev",
        # repo_flow="flux1-dev.sft",
        # repo_ae="ae.sft",
        ckpt_path=None,  # os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "schnell": ModelSpec(
        # repo_id="black-forest-labs/FLUX.1-schnell",
        # repo_flow="flux1-schnell.sft",
        # repo_ae="ae.sft",
        ckpt_path=None,  # os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=None,  # os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    # ==================== FLUX.2 MODELS ====================
    # Flux.2 Klein 9B Base - 9 billion parameters, optimized for consumer hardware
    # Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B
    "flux2_klein_9b": ModelSpec(
        ckpt_path=None,
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,  # Double blocks (will be auto-detected from checkpoint)
            depth_single_blocks=38,  # Single blocks (will be auto-detected from checkpoint)
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,  # Klein supports guidance
        ),
        ae_path=None,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    # Flux.2 Dev - 32 billion parameters, full capacity model
    # Source: https://huggingface.co/black-forest-labs/FLUX.2-dev
    "flux2_dev": ModelSpec(
        ckpt_path=None,
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=4096,  # Larger hidden size for 32B model
            mlp_ratio=4.0,
            num_heads=32,  # More attention heads
            depth=28,  # More double blocks (will be auto-detected)
            depth_single_blocks=56,  # More single blocks (will be auto-detected)
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=None,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

# Model version constants
MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_VERSION_FLUX_V2 = "flux2"
MODEL_VERSION_FLUX2_KLEIN = "flux2_klein"
MODEL_VERSION_FLUX2_DEV = "flux2_dev"

# endregion

# region math


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# endregion


# region layers


# for cpu_offload_checkpointing


def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, (list, tuple)):
        return [to_cuda(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    # def forward(self, x):
    #     if self.training and self.gradient_checkpointing:
    #         def create_custom_forward(func):
    #             def custom_forward(*inputs):
    #                 return func(*inputs)
    #             return custom_forward
    #         return torch.utils.checkpoint.checkpoint(create_custom_forward(self._forward), x, use_reentrant=USE_REENTRANT)
    #     else:
    #         return self._forward(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # return (x * rrms).to(dtype=x_dtype) * self.scale
        return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    # this is not called from DoubleStreamBlock/SingleStreamBlock because they uses attention function directly
    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, mlp_gated: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_gated = mlp_gated
        mlp_in_dim = mlp_hidden_dim * (2 if mlp_gated else 1)
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_in_dim, bias=True),
            nn.GELU(approximate="tanh") if not mlp_gated else nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_in_dim, bias=True),
            nn.GELU(approximate="tanh") if not mlp_gated else nn.Identity(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _run_mlp(self, mlp: nn.Sequential, x: Tensor) -> Tensor:
        mlp_in = mlp[0](x)
        if self.mlp_gated:
            gate, value = mlp_in.chunk(2, dim=-1)
            mlp_act = nn.functional.gelu(gate, approximate="tanh") * value
        else:
            mlp_act = mlp[1](mlp_in)
        return mlp[2](mlp_act)

    def _forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # make attention mask if not None
        attn_mask = None
        if txt_attention_mask is not None:
            # F.scaled_dot_product_attention expects attn_mask to be bool for binary mask
            attn_mask = txt_attention_mask.to(torch.bool)  # b, seq_len
            attn_mask = torch.cat(
                (attn_mask, torch.ones(attn_mask.shape[0], img.shape[1], device=attn_mask.device, dtype=torch.bool)), dim=1
            )  # b, seq_len + img_len

            # broadcast attn_mask to all heads
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self._run_mlp(self.img_mlp, (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self._run_mlp(self.txt_mlp, (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, img, txt, vec, pe, txt_attention_mask, use_reentrant=False)
            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), img, txt, vec, pe, txt_attention_mask, use_reentrant=False
            )

        else:
            return self._forward(img, txt, vec, pe, txt_attention_mask)


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_gated: bool = False,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_gated = mlp_gated
        mlp_in_dim = self.mlp_hidden_dim * (2 if mlp_gated else 1)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_in_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.mlp_gated:
            qkv, mlp_in = torch.split(self.linear1(x_mod), [3 * self.hidden_size, 2 * self.mlp_hidden_dim], dim=-1)
            gate, value = mlp_in.chunk(2, dim=-1)
            mlp = nn.functional.gelu(gate, approximate="tanh") * value
        else:
            qkv, mlp_in = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            mlp = self.mlp_act(mlp_in)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # make attention mask if not None
        attn_mask = None
        if txt_attention_mask is not None:
            # F.scaled_dot_product_attention expects attn_mask to be bool for binary mask
            attn_mask = txt_attention_mask.to(torch.bool)  # b, seq_len
            attn_mask = torch.cat(
                (
                    attn_mask,
                    torch.ones(
                        attn_mask.shape[0], x.shape[1] - txt_attention_mask.shape[1], device=attn_mask.device, dtype=torch.bool
                    ),
                ),
                dim=1,
            )  # b, seq_len + img_len = x_len

            # broadcast attn_mask to all heads
            attn_mask = attn_mask[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)

        # compute attention
        attn = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, mlp), 2))
        return x + mod.gate * output

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None) -> Tensor:
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing:
                return checkpoint(self._forward, x, vec, pe, txt_attention_mask, use_reentrant=False)

            # cpu offload checkpointing

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    cuda_inputs = to_cuda(inputs)
                    outputs = func(*cuda_inputs)
                    return to_cpu(outputs)

                return custom_forward

            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(self._forward), x, vec, pe, txt_attention_mask, use_reentrant=False
            )
        else:
            return self._forward(x, vec, pe, txt_attention_mask)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


# endregion


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    mlp_gated=params.mlp_gated,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, mlp_gated=params.mlp_gated)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.blocks_to_swap = None

        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload

        self.time_in.enable_gradient_checkpointing()
        self.vector_in.enable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.enable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(cpu_offload=cpu_offload)

        print(f"FLUX: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False

        self.time_in.disable_gradient_checkpointing()
        self.vector_in.disable_gradient_checkpointing()
        if self.guidance_in.__class__ != nn.Identity:
            self.guidance_in.disable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("FLUX: Gradient checkpointing disabled.")

    def get_block_swap_limits(self) -> tuple[int, int, int]:
        max_double_blocks = max(0, self.num_double_blocks - 2)
        max_single_blocks = max(0, self.num_single_blocks - 2)

        max_combined = 0
        upper_bound = max(0, max_double_blocks * 2 + 2)
        for candidate in range(upper_bound + 1):
            candidate_double = candidate // 2
            candidate_single = (candidate - candidate_double) * 2
            if candidate_double <= max_double_blocks and candidate_single <= max_single_blocks:
                max_combined = candidate

        return max_double_blocks, max_single_blocks, max_combined

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        if num_blocks is None or num_blocks <= 0:
            self.blocks_to_swap = 0
            self.offloader_double = None
            self.offloader_single = None
            print("FLUX: Block swap disabled.")
            return

        max_double_blocks, max_single_blocks, max_combined = self.get_block_swap_limits()
        if num_blocks > max_combined:
            raise ValueError(
                f"Cannot swap more than {max_combined} combined blocks for this model "
                f"(double<= {max_double_blocks}, single<= {max_single_blocks}). Requested: {num_blocks}."
            )

        self.blocks_to_swap = num_blocks
        double_blocks_to_swap = num_blocks // 2
        single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2

        self.offloader_double = ModelOffloader(
            self.double_blocks, self.num_double_blocks, double_blocks_to_swap, device  # , debug=True
        )
        self.offloader_single = ModelOffloader(
            self.single_blocks, self.num_single_blocks, single_blocks_to_swap, device  # , debug=True
        )
        print(
            f"FLUX: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        save_double_blocks = None
        save_single_blocks = None
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            save_single_blocks = self.single_blocks
            self.double_blocks = None
            self.single_blocks = None

            # swapped blocks are detached from self and won't be included in self.named_parameters()
            swap_meta_params, swap_meta_buffers = self.materialize_meta_tensors_in_modules(
                list(save_double_blocks) + list(save_single_blocks), torch.device("cpu")
            )
            if swap_meta_params or swap_meta_buffers:
                print(
                    f"FLUX: Materialized meta tensors in swap blocks (params={swap_meta_params}, buffers={swap_meta_buffers})"
                )

        meta_params, meta_buffers = self.materialize_meta_tensors(torch.device("cpu"))
        if meta_params or meta_buffers:
            print(
                f"FLUX: Materialized meta tensors before move (params={meta_params}, buffers={meta_buffers})"
            )

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = save_double_blocks
            self.single_blocks = save_single_blocks

    def _materialize_meta_tensors_for_module(self, root_module: nn.Module, device: torch.device) -> tuple[int, int]:
        float8_dtypes = {
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        }

        def _init_floating_tensor(shape, target_dtype):
            init_dtype = torch.float32 if target_dtype in float8_dtypes else target_dtype
            tensor = torch.empty(shape, dtype=init_dtype, device=device)
            if tensor.ndim >= 2:
                nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
            else:
                nn.init.zeros_(tensor)
            if init_dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)
            return tensor

        meta_params = 0
        meta_buffers = 0

        for module in root_module.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None or not getattr(parameter, "is_meta", False):
                    continue

                if parameter.is_floating_point():
                    new_data = _init_floating_tensor(parameter.shape, parameter.dtype)
                else:
                    new_data = torch.empty(parameter.shape, dtype=parameter.dtype, device=device)
                    new_data.zero_()

                module._parameters[name] = nn.Parameter(new_data, requires_grad=parameter.requires_grad)
                meta_params += 1

            for name, buffer in list(module._buffers.items()):
                if buffer is None or not getattr(buffer, "is_meta", False):
                    continue

                new_buffer = torch.empty(buffer.shape, dtype=buffer.dtype, device=device)
                new_buffer.zero_()
                module._buffers[name] = new_buffer
                meta_buffers += 1

        return meta_params, meta_buffers

    def materialize_meta_tensors(self, device: torch.device) -> tuple[int, int]:
        return self._materialize_meta_tensors_for_module(self, device)

    def materialize_meta_tensors_in_modules(self, modules: list[nn.Module], device: torch.device) -> tuple[int, int]:
        total_params = 0
        total_buffers = 0
        for module in modules:
            params, buffers = self._materialize_meta_tensors_for_module(module, device)
            total_params += params
            total_buffers += buffers
        return total_params, total_buffers

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # safety net: ensure swap blocks have no meta tensors before offloader tries moving them
        swap_meta_params, swap_meta_buffers = self.materialize_meta_tensors_in_modules(
            list(self.double_blocks) + list(self.single_blocks), torch.device("cpu")
        )
        if swap_meta_params or swap_meta_buffers:
            print(
                f"FLUX: Materialized meta tensors before swap prepare (params={swap_meta_params}, buffers={swap_meta_buffers})"
            )

        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        txt_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if not self.blocks_to_swap:
            for block in self.double_blocks:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
            img = torch.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)
        else:
            for block_idx, block in enumerate(self.double_blocks):
                self.offloader_double.wait_for_block(block_idx)

                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)

                self.offloader_double.submit_move_blocks(self.double_blocks, block_idx)

            img = torch.cat((txt, img), 1)

            for block_idx, block in enumerate(self.single_blocks):
                self.offloader_single.wait_for_block(block_idx)

                img = block(img, vec=vec, pe=pe, txt_attention_mask=txt_attention_mask)

                self.offloader_single.submit_move_blocks(self.single_blocks, block_idx)

        img = img[:, txt.shape[1] :, ...]

        if self.training and self.cpu_offload_checkpointing:
            img = img.to(self.device)
            vec = vec.to(self.device)

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        return img
