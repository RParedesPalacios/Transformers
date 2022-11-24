
from ast import main
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from PatchEmbedding import PatchEmbedding
from torch.nn import MultiheadAttention



class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, heads, drop_p=0.1, forward_expansion=4, forward_drop_p=0.0):
        super().__init__()
        self.attention = MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * d_model, d_model),
        )

    def forward(self, x, pad_mask=None):
        output,_= self.attention(x,x,x,key_padding_mask=pad_mask)
        output = x + self.dropout1(output)
        output = self.norm1(output)  


        output2 = self.feed_forward(output) 
        output = self.norm2(output + self.dropout2(output2))

        return output


import copy
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, patch_num, patch_dim, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = PatchEmbedding(d_model, patch_num, patch_dim)
        self.layers = get_clones(TransformerEncoderBlock(d_model, heads), N)
    def forward(self, src, pad_mask=None):
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x,pad_mask)
        return x




if __name__ == "__main__":
    x=torch.randn(10,64,2048)   
    pad=torch.randint(0,1,(10,64))
    print("...........")
    TE=TransformerEncoder(patch_num= 64, patch_dim=2048, d_model=768, N=6, heads=8)
    te=TE(x,pad) 
    print("TE=",te.shape)
