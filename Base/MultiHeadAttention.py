
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


###########################################
'''
from PatchEmbedding import PatchEmbedding

x=torch.randn(10,64,2048)

pe=PatchEmbedding(emb_size=768, patch_num=64, bdim=2048)
patches_embedded = pe(x)
print("PE=",patches_embedded.shape)

print("...........")
MA=MultiHeadAttention(emb_size=768,num_heads=8)
ma=MA(patches_embedded) 
print("MA=",ma.shape)

'''
