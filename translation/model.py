"""
ArthSathi Seq2Seq Translation Model — built from scratch.

Architecture: Transformer encoder-decoder (~60M params at default config).
Designed to run on CPU / Colab free tier.

Member C owns this file.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        def split(x):
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(self.q(query)), split(self.k(key)), split(self.v(value))

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))

        x = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class Seq2SeqTranslator(nn.Module):
    """
    Small transformer seq2seq model for Hindi↔English and Marathi↔English translation.

    Default config (~60M params):
        d_model=512, n_heads=8, n_enc=6, n_dec=6, d_ff=2048, vocab=32000

    Lighter CPU config (~15M params):
        d_model=256, n_heads=4, n_enc=4, n_dec=4, d_ff=1024, vocab=16000
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,       # use 256 for CPU; 512 for Colab GPU
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    @staticmethod
    def make_causal_mask(sz: int, device) -> torch.Tensor:
        """Upper-triangular mask to prevent decoder attending to future tokens."""
        return torch.tril(torch.ones(sz, sz, device=device)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def make_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
