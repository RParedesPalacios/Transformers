import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model, heads, drop_p=0.1, forward_expansion=4, forward_drop_p=0.0):
        super().__init__()
        self.attention1 = MultiheadAttention(d_model, heads, batch_first=True)
        self.attention2 = MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)
        self.dropout3 = nn.Dropout(drop_p)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.GELU(),
            nn.Dropout(forward_drop_p),
            nn.Linear(forward_expansion * d_model, d_model),
        )

    def forward(self, x, e_outputs, trg_mask, pad_mask):
        output,_= self.attention1(x,x,x, attn_mask=trg_mask, key_padding_mask=pad_mask)
        output = x + self.dropout1(output)
        output = self.norm1(output)  

        output2,_ = self.attention2(output, e_outputs, e_outputs)
        output = output + self.dropout2(output2)
        output = self.norm2(output)

        output2 = self.feed_forward(output) 
        output = self.norm3(output + self.dropout3(output2))

        return output


class TransfomerDecoder(nn.Module):
    def __init__(self, trg_vocab, trg_lenght, d_model, N, heads):
        super().__init__()
        self.trg_vocab = trg_vocab
        self.d_model = d_model
        self.N = N
        self.heads = heads

        self.cptn_emb = nn.Embedding(trg_vocab, d_model)
        self.pos_emb = PositionalEncoding(d_model, trg_lenght)

        self.blocks = nn.ModuleList([TransformerDecoderBlock(d_model, heads) for _ in range(N)])
        self.out = nn.Linear(d_model, trg_vocab)

    def get_attn_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Pytorch MultiHeadAttention:
        For a float mask, the mask values will be added to the attention weight.
        0 -inf -inf
        0 0 -inf
        0 0 0
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, trg, e_outputs, pad_mask=None):
        # create masks, then pass to decoder

        attn_mask = self.get_attn_subsequent_mask(trg.size()[1])

        # pad_mask: [batch_size, trg_len]
        # True: ignore this position

        x = trg
        x = self.cptn_emb(x)
        x = self.pos_emb(x)
        for i in range(self.N):
            x = self.blocks[i](x, e_outputs, attn_mask, pad_mask)
        x = self.out(x)
        return x


