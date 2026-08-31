"""
ArthSathi Small Language Model — GPT-style decoder-only transformer.
~15M params at default config (CPU/Colab free friendly).
Trained on scheme + insurance + legal domain text in Hindi+English.

Member A owns this file.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask — registered as buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.d_k).transpose(1, 2) for t in qkv]

        scale = math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ArthSathiLM(nn.Module):
    """
    Small autoregressive language model for scheme + insurance domain.

    Default config ~15M params (runs on CPU):
        vocab=16000, d_model=256, n_heads=4, n_layers=6, max_len=512

    Larger config ~117M params (use on Colab GPU):
        vocab=16000, d_model=768, n_heads=12, n_layers=12, max_len=512
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: input embedding = output projection (reduces params, improves quality)
        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_embed(idx) + self.pos_embed(positions))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        """Autoregressive generation with top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to max_len
            ctx = prompt_ids[:, -self.max_len:]
            logits = self(ctx)[:, -1, :]  # last token logits

            logits = logits / temperature
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            prompt_ids = torch.cat([prompt_ids, next_id], dim=1)

        return prompt_ids

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
