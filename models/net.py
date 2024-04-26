import torch
import torch.nn as nn
from torch import nn
from collections import OrderedDict
import torch
from torch import nn
from clip.clip import *
from torch.nn.utils import weight_norm

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class BatchNorm(nn.BatchNorm1d):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = x.permute(1,2,0) #LND->NDL
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type).permute(2,0,1)#NDL->LND
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        # self.ln_1 = BatchNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.ln_2 = BatchNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    def get_attention(self, x: torch.Tensor):
        return self.resblocks.get_attention(x)

class backbone(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, dropout: int):
        super().__init__()
        self.input_resolution = input_resolution
        scale = width ** -0.5
        self.conv1 = nn.Parameter(scale * torch.randn(patch_size, width))
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(input_resolution + 1, width))
        self.ln_pre = LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads,dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers=layers)
        self.transformer = Transformer(width, layers, heads)

    def random_masking(self, x, mask_ratio):

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio = 0.0):
        x = x @ self.conv1
        x = x + self.positional_embedding.to(x.dtype)[1:,:]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.class_embedding.to(x.dtype) + self.positional_embedding.to(x.dtype)[0] + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) 
        # x = x.permute(0, 2, 1)
        # x = self.conv1(x)
        # x = x.permute(0, 2, 1)
        a = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # N * L * D
        # x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    
class TimeTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, dropout: int):
        super().__init__()
        self.backbone = backbone(input_resolution, patch_size, width, layers, heads, dropout)
        self.output_dim = output_dim
        scale = width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj2 = nn.Parameter(scale * torch.randn(width, 512))

    def forward(self, x: torch.Tensor, mask_ratio = 0.0):
        x = self.backbone(x, mask_ratio)
        x = x[:, 0, :]
        return x, x @ self.proj, x @ self.proj2

    
    
    

