import math
from functools import partial
from typing import List, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            attn = attn.softmax(dim=-1)
            x = (attn @ v)

        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(self, x, block_idx, mask):
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    """
    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule
    
    def forward(self, x):
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension
            
            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B = B * math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


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


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
    x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def conv_nd(n):
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


class PatchEmbed(nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class HieraEncoder(nn.Module):
    def __init__(self,
        input_size = (224, 224),
        in_chans = 3,
        embed_dim = 96,  # initial embed dim
        num_heads = 1,  # initial number of heads
        num_classes = 1000,
        stages = (2, 3, 16, 3),
        norm_layer = "LayerNorm",
        patch_stride = (4, 4),
        mask_unit_size  = (8, 8),  # must divide q_stride ** (#stages-1)
        q_pool = 3,  # number of q_pool stages
        q_stride = (2, 2),
        patch_kernel = (7, 7),
        patch_padding = (3, 3),
        sep_pos_embed = False,
        drop_path_rate = 0.0,
        mask_unit_attn = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        head_dropout: float = 0.0
    ):
        super().__init__()
        # get layer norm
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        # argument settings
        depth = sum(stages)
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.sep_pos_embed = sep_pos_embed

        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        # patch embedding
        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )
        
        # positional embedding
        if sep_pos_embed: # for video I guess.
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for idx in range(depth):
            dim_out = embed_dim
            
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if idx - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if idx in q_pool_blocks:
                    flat_mu_size = flat_mu_size // flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if idx in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

            self.norm = norm_layer(embed_dim)
            # projection layer.
            self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)
        

        # Modules for Masked HieraEncoder 
        encoder_dim_out = self.blocks[-1].dim_out 
        self.encoder_norm = norm_layer(encoder_dim_out)

        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]

        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform
        # --------------------------------------------------------------------------

        # Initialize all parameters
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)
        
    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_pos_embed(self):
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed
    
    def forward(self, x, mask, return_intermediates=False):
        if isinstance(x, list):
            x = x[0]
        intermediates = []

        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )
        
        # MLP (FFN)
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))
        
        if mask is None:
            x = x.mean(dim=1)
            x = self.norm(x)
            x = self.head(x)

        # without MAE training
        #if return_intermediates:
        #    return x, intermediates
        
        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        # Fuse multi-scale intermediate outputs
        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            x += apply_fusion_head(head, interm_x)

        x = self.encoder_norm(x)

        return x, mask

        
