
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from PatchEmbedding import PatchEmbedding
from MultiHeadAttention import MultiHeadAttention


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,emb_size, heads,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, heads),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Module):
    def __init__(self, patch_num, patch_dim, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = PatchEmbedding(d_model, patch_num, patch_dim)
        self.layers = get_clones(TransformerEncoderBlock(d_model, heads), N)
    def forward(self, src):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x)
        return x


'''
x=torch.randn(10,64,2048)

print("...........")
TE=TransformerEncoder(patch_num= 64, patch_dim=2048, d_model=768, N=6, heads=8)
te=TE(x) 
print("TE=",te.shape)
''' 
