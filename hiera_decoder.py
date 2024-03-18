from functools import partial
from typing import Tuple, Optional

import math
import torch
import torch.nn as nn


def undo_windowing(
    x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0, # Ratio of mlp hidden dim to embedding dim
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        # perform either global attention or mask unit attention using MaskUnitAttention module
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(x):
        # Attention + Q Pooling
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class MAEHieraDecoder(nn.Module):
    def __init__(self,
        in_chans = 3,
        q_pool = 0, 
        tokens_spatial_shape = 0, 
        q_stride = 0,
        mlp_ratio = 4.0,
        decoder_embed_dim = 512,
        decoder_depth = 8,
        decoder_num_heads = 16,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        patch_stride = (4, 4),
        **kwdargs):
        super().__init__(
            encoder_dim_out=encoder_dim_out,
            decoder_embed_dim=decoder_embed_dim,
            #patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )
        self.q_pool = q_pool
        self.tokens_spatial_shape = tokens_spatial_shape
        self.q_stride = q_stride
        self.decoder_depth = decoder_depth

        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # math.prod(self.tokens_spatial_shape_final) ==> count total number of tokens.
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(self.decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # patch stride of prediction
        self.pred_stride = patch_stride[-1] * (self.q_stride[-1] ** self.q_pool)

        # predictor
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]::
        """
        x: visual patches
        x_dec: visual patches + maksed patches
        Note: Mask = 1 --> visual patches 0 --> masked patches...
        """
        x = self.self.decoder_embed(x)

        # combine visual and mask tokens
        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, all numbers of mask units]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec # combine visual tokens with masked tokens.
        
        # convert x and mask to spatial order (e.g., tokens)
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # project outputs of blocks to Predictor
        x = self.decoder_pred(x)

        return x, mask

