"""Sıfırdan bayt düzeyinde BPE tokenizer (Orion'a ait; hazır sözlük yok).

Kimlik düzeni:
  0..7      özel işaretler (<pad> <bos> <eos> <sep> <t2c> <plan> <ans> <unk>)
  8..263    256 bayt
  264..     birleştirmeler (merges) sırasıyla
Ön bölme: harf dizileri, TEK TEK rakamlar (saat/sayı kopyalamayı kolaylaştırır),
noktalama dizileri ve boşluklar. Tur 3: kelime sonundaki yaygın Türkçe ekler ayrı parça olur
("ödemeyi" → "ödeme" + "yi", "telefonda" → "telefon" + "da"); böylece model kökü girdiden
kopyalayıp eki değiştirebilir (ör. mastar: "ödeme" + "k"). Bölme kayıpsızdır (parçalar boşluksuz birleşir). Kotlin karşılığı: orion-core/.../model/BpeTokenizer.kt
(aynı düzenli ifade; Java ve Python `regex` aynı bölmeyi verir, test vektörleriyle sınanır).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<t2c>", "<plan>", "<ans>", "<unk>"]
PAD, BOS, EOS, SEP, T2C, PLAN, ANS, UNK = range(8)
TASK_TOKEN = {"t2c": T2C, "plan": PLAN, "ans": ANS}
BYTE_OFFSET = len(SPECIALS)
FIRST_MERGE = BYTE_OFFSET + 256

# Ayrılan ekler: çoğul/iyelik, bulunma/ayrılma, belirtme/yönelme (kaynaştırmalı) ve -ile.
SUFFIXES = ("larımı|lerimi|larını|lerini|larım|lerim|ların|lerin|ları|leri|lar|ler|dan|den|tan|ten|"
            "da|de|ta|te|yla|yle|yı|yi|yu|yü|mı|mi|mu|mü|ya|ye|nı|ni|nu|nü")
PRETOKENIZE = (r" ?\p{L}{3,}?(?=(?:" + SUFFIXES + r")(?!\p{L}))| ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
LEGACY_PRETOKENIZE = r" ?\p{L}+| ?\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_PRE = re.compile(PRETOKENIZE)


def pretokenize(text: str, pattern=None) -> list[str]:
    return (pattern or _PRE).findall(text)


class BPE:
    def __init__(self, merges: list[tuple[int, int]], pattern: str = PRETOKENIZE):
        self.pattern = pattern
        self._pre = re.compile(pattern)
        self.merges = [tuple(m) for m in merges]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {BYTE_OFFSET + b: bytes([b]) for b in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[FIRST_MERGE + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE + len(self.merges)

    # ------------------------------------------------------------------ eğitim
    @classmethod
    def train(cls, texts, vocab_size: int = 4096, min_freq: int = 2) -> "BPE":
        word_freq: Counter[str] = Counter()
        for t in texts:
            word_freq.update(pretokenize(t))
        words = [[BYTE_OFFSET + b for b in w.encode("utf-8")] for w in word_freq]
        freqs = list(word_freq.values())
        pair_count: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += freqs[wi]
                where[p].add(wi)
        merges: list[tuple[int, int]] = []
        while FIRST_MERGE + len(merges) < vocab_size and pair_count:
            # en sık çift; eşitlikte en küçük kimlik çifti (deterministik)
            best, cnt = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt < min_freq:
                break
            new_id = FIRST_MERGE + len(merges)
            merges.append(best)
            for wi in list(where.pop(best, ())):
                w = words[wi]
                f = freqs[wi]
                for p in zip(w, w[1:]):
                    pair_count[p] -= f
                    if pair_count[p] <= 0:
                        del pair_count[p]
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_count[p] += f
                    where[p].add(wi)
            pair_count.pop(best, None)
        return cls(merges)

    # ------------------------------------------------------------------ kodlama
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = [BYTE_OFFSET + b for b in chunk.encode("utf-8")]
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids = ids[:best_i] + [FIRST_MERGE + best_rank] + ids[best_i + 2:]
        if len(self._cache) < getattr(self, "_cache_limit", 200_000):
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for c in self._pre.findall(text):
            out.extend(self._encode_chunk(c))
        return out

    def decode(self, ids) -> str:
        return b"".join(self.vocab.get(i, b"") for i in ids if i >= BYTE_OFFSET).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ dosya
    def to_json(self) -> dict:
        return {"format": "orion-bpe", "version": 1, "specials": SPECIALS, "byte_offset": BYTE_OFFSET,
                "pretokenize": self.pattern, "merges": [list(m) for m in self.merges],
                # copy.py: modelin sözlüğü = bpe + bu kadar işaretçi (girdiden kopyalama)
                "pointers": 96}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        assert d["format"] == "orion-bpe" and d["byte_offset"] == BYTE_OFFSET
        return cls([tuple(m) for m in d["merges"]], d.get("pretokenize", LEGACY_PRETOKENIZE))
