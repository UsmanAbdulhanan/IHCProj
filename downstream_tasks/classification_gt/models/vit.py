""" Vision Transformer (ViT) in PyTorch
"""
import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, num_hidden_layers=1, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()

        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features),
                    nn.GELU(),
                    nn.Dropout(drop)
                ) for _ in range(1, num_hidden_layers)
            ]
        )

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # A = Q*K^T
        self.equation1 = 'bhid,bhjd->bhij'
        # attn = A*V
        self.equation2 = 'bhij,bhjd->bhid'

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        dots = torch.einsum(self.equation1, q, k) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)

        out = torch.einsum(self.equation2, attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x1, x2 = [x, x]
        x = torch.add(x1, self.attn(self.norm1(x2)))
        x1, x2 = [x, x]
        x = torch.add(x1, self.mlp(self.norm2(x2)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            self,
            embed_dim=64, num_heads=8, depth=3, mlp_ratio=2.,
            qkv_bias=False, proj_drop_rate=0., attn_drop_rate=0.,
            **kwargs
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=proj_drop_rate, attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = torch.index_select(x, dim=1, index=torch.tensor(0, device=x.device))
        x = x.squeeze(1)

        # embedding
        return x
