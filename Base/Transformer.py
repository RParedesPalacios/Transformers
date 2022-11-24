
import torch
from torch import nn

from ViTEncoder2 import TransformerEncoder
from NLPDecoder import TransfomerDecoder


class Transformer(nn.Module):
    def __init__(self, patch_num, patch_dim, d_model, trg_vocab, trg_lenght, N, heads):
        super().__init__()
        self.encoder = TransformerEncoder(patch_num, patch_dim, d_model, N, heads)
        self.decoder = TransfomerDecoder(trg_vocab, trg_lenght, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_pad=None, trg_pad=None):
        e_outputs = self.encoder(src,src_pad)
        d_output = self.decoder(trg, e_outputs)
        return d_output


if __name__ == "__main__":
    batch_size=32
    patch_num=64
    patch_dim=2048
    trg_length=25
    d_model=768
    trg_vocab=100
    N=6
    heads=8
    
    src = torch.rand(batch_size, patch_num, patch_dim)
    trg = torch.randint(0, trg_vocab , (batch_size, trg_length), dtype=torch.long)

    src_pad = torch.randint(0, 1, (batch_size, patch_num))
    trg_pad = torch.randint(0, 1, (batch_size, trg_length))
    
    print("src=",src.shape)
    print("trg=",trg.shape)
    model = Transformer(patch_num, patch_dim, d_model, trg_vocab, trg_length, N, heads)
    output = model(src, trg, src_pad, trg_pad)
    print("out=",output.shape)
