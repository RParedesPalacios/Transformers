
import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange

###########################################
# Patch Empedding                         #
#                                         #
# Input: (b,patches,dim)                  #
# patches have been extracted  and        #
# projected with backbone model           #
#                                         #
# Output: (b,patches,emb_size)            #
# project to emb_size                     #
# and add position embedding              #
###########################################
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()

        self.conv=nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.arrange=Rearrange('b e (h) (w) -> b (h w) e')
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.arrange(x)
        return x
  
'''
pe=PatchEmbedding()
print(pe(torch.randn(10,3,64,64)).shape)
'''