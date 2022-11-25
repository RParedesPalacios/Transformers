
import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange

###########################################
# Patch Empedding                         #
#                                         #
# Input: (b,c,h,w)                        #
#                                         #
# Output: (b,patches,emb_size)            #
# patches are extracted  and              #
# projected to emb_size with a conv2d     #
# and add position embedding              #
###########################################
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, patch_num, emb_size):
        self.patch_size = patch_size
        self.patch_num = patch_num
        super().__init__()

        self.conv=nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.arrange=Rearrange('b e (h) (w) -> b (h w) e')
        self.positions = nn.Parameter(torch.randn(patch_num, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.arrange(x)
        x = x + self.positions
        return x
  
'''
pe=PatchEmbedding()
print(pe(torch.randn(10,3,64,64)).shape)
'''