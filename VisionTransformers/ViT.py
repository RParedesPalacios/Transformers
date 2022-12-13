
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from PatchEmbedding import PatchEmbedding
from torch.nn import MultiheadAttention


class TransformerBlock(nn.Module):

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

class ViT(nn.Module):
    def __init__(self, img_size, img_channels, patch_size, d_model, N, heads, num_classes):
        super().__init__()
        self.img_size=img_size
        self.img_channels=img_channels
        self.patch_size=patch_size
        self.d_model=d_model
        self.N = N

        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert d_model % heads == 0, 'Transformer dim must be divisible by the number of heads.'
        
        self.patch_num=(img_size//patch_size)**2

        print("=============================================")
        print("Image size %d x %d" %(img_size,img_size))
        print("Building ViT with %d patches of %d x %d" %(self.patch_num,patch_size,patch_size))
        print("Transformer block dim",self.d_model)
        print("=============================================")

        self.embed = PatchEmbedding(img_channels, patch_size, self.patch_num, d_model)
        self.layers = get_clones(TransformerBlock(d_model, heads), N)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, src, pad_mask=None):
        # x = (batch, channels, height, width)
        x = self.embed(src)
        for i in range(self.N):
            x = self.layers[i](x,pad_mask)
        x = x.mean(dim = 1) 
        x = self.mlp_head(x)
        return x




if __name__ == "__main__":
    x=torch.randn(10,3,255,256)   
    TE=ViT(img_size=256, img_channels=3, patch_size= 16, d_model=768, N=6, heads=8,num_classes=1024)
    te=TE(x) 
    print("TE=",te.shape)
