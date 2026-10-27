"""Çok küçük GPT (1-5 milyon parametre). Dünya bilgisi yok: yalnızca komut ayrıştırma,
araç/adım planı ve sabit biçimli cevap öğrenir.

Kotlin çalıştırıcısı (orion-core/.../model/TinyGpt.kt) aynı hesabı yapar:
ön-LayerNorm, öğrenilen konum gömmesi, nedensel çok başlı dikkat, tanh-yaklaşımlı GELU,
giriş/çıkış gömmesi paylaşımlı.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GptConfig:
    vocab_size: int = 4096
    ctx: int = 192
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 768
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, c: GptConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_ff)
        self.fc2 = nn.Linear(c.d_ff, c.d_model)
        self.n_head = c.n_head
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        h = self.n_head
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh")))
        return x

    def forward_kv(self, x: torch.Tensor, past):
        """Önbellekli adım: past=(k,v) ya da None. Dönen: (x, (k,v))."""
        b, t, d = x.shape
        h = self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(b, t, h, d // h).transpose(1, 2)
        k = k.view(b, t, h, d // h).transpose(1, 2)
        v = v.view(b, t, h, d // h).transpose(1, 2)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))
        return x, (k, v)


class TinyGpt(nn.Module):
    def __init__(self, c: GptConfig):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.d_model)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok.weight.T  # paylaşımlı çıkış

    @torch.no_grad()
    def greedy(self, prompt: list[int], eos: int, max_new: int = 96) -> list[int]:
        self.eval()
        ids = list(prompt)
        for _ in range(max_new):
            if len(ids) >= self.config.ctx:
                break
            logits = self(torch.tensor([ids]))[0, -1]
            nxt = int(torch.argmax(logits))
            if nxt == eos:
                break
            ids.append(nxt)
        return ids[len(prompt):]

    @torch.no_grad()
    def forward_kv(self, idx: torch.Tensor, past=None):
        t0 = 0 if past is None else past[0][0].shape[2]
        x = self.tok(idx) + self.pos(torch.arange(t0, t0 + idx.shape[1], device=idx.device))
        new = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk.forward_kv(x, None if past is None else past[i])
            new.append(kv)
        return self.ln_f(x) @ self.tok.weight.T, new

    @torch.no_grad()
    def greedy_batch(self, prompts: list[list[int]], eos: int, pad: int, max_new: int = 96, batch: int = 64,
                     with_conf: bool = False):
        """Toplu açgözlü çözümleme: aynı uzunluktaki istemler birlikte, KV önbelleğiyle (dolgu yok).
        with_conf=True: her çıktı için güven = seçilen belirteçlerin en düşük olasılığı (EOS dahil)."""
        self.eval()
        outs: list[list[int]] = [[] for _ in prompts]
        conf: list[float] = [1.0 for _ in prompts]
        by_len: dict[int, list[int]] = {}
        for i, p in enumerate(prompts):
            by_len.setdefault(len(p), []).append(i)
        for length, idxs in by_len.items():
            for s in range(0, len(idxs), batch):
                group = idxs[s:s + batch]
                x = torch.tensor([prompts[i] for i in group])
                logits, past = self.forward_kv(x)
                done = [False] * len(group)
                for step in range(max_new):
                    if length + step >= self.config.ctx:
                        break
                    probs = torch.softmax(logits[:, -1].float(), dim=-1)
                    pmax, nxt = probs.max(dim=-1)
                    for r, i in enumerate(group):
                        if not done[r]:
                            conf[i] = min(conf[i], float(pmax[r]))
                            if int(nxt[r]) == eos:
                                done[r] = True
                            else:
                                outs[i].append(int(nxt[r]))
                    if all(done) or length + step + 1 >= self.config.ctx:
                        break
                    logits, past = self.forward_kv(nxt[:, None], past)
        return (outs, conf) if with_conf else outs
