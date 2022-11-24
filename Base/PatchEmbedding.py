
import torch
from torch import nn
from torch import Tensor



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
    def __init__(self, emb_size, patch_num, patch_dim):
        super().__init__()
        self.emb_size = emb_size
        self.patch_dim = patch_dim
        self.patch_num = patch_num
        
        # projection from backbone to embedding size
        self.projection = nn.Linear(patch_dim, emb_size)
        self.positions = nn.Parameter(torch.randn(patch_num, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, p, d = x.shape

        x = self.projection(x)
        x += self.positions
        return x
    
'''
pe=PatchEmbedding(768, 64, 2048)
print(pe(torch.randn(10,64,2048)).shape)
'''
