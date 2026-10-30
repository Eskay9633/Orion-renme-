"""Çekirdek modeli eğitir. GitHub Actions CPU'su için tasarlandı:
- --time-budget-min ile süre sınırı (iş başına <= 6 saat), checkpoint ile kaldığı yerden devam
- veri her adımda dilbilgisinden üretilir (sonsuz akış) + varsa damıtılmış örnekler
- --teacher verilirse BİLGİ DAMITMA: öğretmen modelin yumuşak olasılıklarıyla KL kaybı

Kullanım:
  python -m orion_core.train --out work/run --steps 6000 --time-budget-min 300
  python -m orion_core.train --out work/student --teacher work/teacher/ckpt.pt --kd-alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import SHARED, datagen
from .bpe import BOS, EOS, PAD, SEP, TASK_TOKEN, BPE
from .copy import N_PTR, encode_target
from .model import GptConfig, TinyGpt


def encode_example(bpe: BPE, ex: dict, ctx: int) -> tuple[list[int], int] | None:
    """[BOS, görev, kaynak..., SEP, hedef..., EOS]; kayıp yalnızca hedef üzerinde.
    Döner: (kimlikler, hedefin başladığı konum)."""
    src = bpe.encode(ex["src"])
    if len(src) > N_PTR:
        return None
    tgt = encode_target(bpe, src, ex["tgt"], ex["task"])
    ids = [BOS, TASK_TOKEN[ex["task"]]] + src + [SEP]
    start = len(ids)
    ids += tgt + [EOS]
    if len(ids) > ctx:
        return None
    return ids, start


def prompt_ids(bpe: BPE, task: str, src: str) -> list[int]:
    return [BOS, TASK_TOKEN[task]] + bpe.encode(src) + [SEP]


def load_distilled(paths: list[str]) -> list[dict]:
    """Damıtılmış örnekler: yalnızca training_allowed=true olanlar (filtre burada ve pipeline'da)."""
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("training_allowed") is True and r.get("task") in TASK_TOKEN:
                out.append({"task": r["task"], "src": r["src"], "tgt": r["tgt"]})
    return out


SEED_SET = SHARED / "core-model" / "seed-train.jsonl"


class Stream:
    def __init__(self, bpe: BPE, ctx: int, seed: int, distilled: list[dict], distill_frac: float,
                 seed_rows: list[dict] | None = None, seed_frac: float = 0.03,
                 ext_rows: list[dict] | None = None, ext_frac: float = 0.0):
        self.bpe, self.ctx = bpe, ctx
        self.rng = random.Random(seed)
        self.distilled = distilled
        self.distill_frac = distill_frac if distilled else 0.0
        self.seed_rows = seed_rows or []
        self.seed_frac = seed_frac if self.seed_rows else 0.0
        # Gerçek Türkçe cümleler (MASSIVE train; massive.py). Yoksa akış eskisiyle birebir aynı (rng tüketilmez).
        from .massive import ExtSampler
        self.ext = ExtSampler(ext_rows or [])
        self.ext_frac = ext_frac if self.ext else 0.0

    def example(self) -> dict:
        if self.seed_frac and self.rng.random() < self.seed_frac:
            return self.rng.choice(self.seed_rows)
        if self.distill_frac and self.rng.random() < self.distill_frac:
            return self.rng.choice(self.distilled)
        if self.ext_frac and self.rng.random() < self.ext_frac:
            return self.ext.sample(self.rng)
        r = self.rng.random()
        if r < 0.8:
            return datagen.t2c_example(self.rng, "train")
        if r < 0.92:
            return datagen.plan_example(self.rng, "train")
        return datagen.answer_example(self.rng, "train")

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        seqs = []
        while len(seqs) < n:
            enc = encode_example(self.bpe, self.example(), self.ctx)
            if enc:
                seqs.append(enc)
        L = max(len(s) for s, _ in seqs)
        x = torch.full((n, L - 1), PAD, dtype=torch.long)
        y = torch.full((n, L - 1), -100, dtype=torch.long)
        for i, (s, start) in enumerate(seqs):
            t = torch.tensor(s)
            x[i, : len(s) - 1] = t[:-1]
            tgt = t[1:].clone()
            tgt[: start - 1] = -100  # kaynak kısmında kayıp yok
            y[i, : len(s) - 1] = tgt
        return x, y


def train_tokenizer(out: Path, vocab_size: int, seed: int, n: int = 60000, extra_texts: list[str] | None = None) -> BPE:
    from .lexicon import lexicon_sentences
    ex = datagen.generate(n, "train", seed=seed)
    texts = [e["src"] for e in ex] + [e["tgt"] for e in ex] + lexicon_sentences(n // 2, seed) + list(extra_texts or [])
    bpe = BPE.train(texts, vocab_size=vocab_size)
    bpe.save(out / "tokenizer.json")
    return bpe


def lr_at(step: int, total: int, base: float, warmup: int = 200) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--time-budget-min", type=float, default=330)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer", help="var olan tokenizer.json (öğretmen ve öğrenci aynı tokenizer'ı kullanmalı)")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--teacher", help="damıtma için öğretmen checkpoint")
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="KL kaybının ağırlığı")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--distilled", nargs="*", default=[], help="damıtılmış jsonl dosyaları")
    ap.add_argument("--distill-frac", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ext-data", help="massive.py prepare çıktısı (MASSIVE train); verilirse eksik/sızıntılı veri = hata")
    ap.add_argument("--ext-frac", type=float, default=None, help="akıştaki dış veri payı (varsayılan massive.EXT_FRAC)")
    a = ap.parse_args(argv)

    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    ext_rows: list[dict] = []
    if a.ext_data:
        from . import massive
        ext_rows = massive.load_train(Path(a.ext_data))  # dosya yoksa ya da değerlendirmeyle çakışırsa DataError
        print(f"dış veri (MASSIVE tr-TR train): {len(ext_rows)} satır", flush=True)
    if a.tokenizer:
        bpe = BPE.load(Path(a.tokenizer))
        bpe.save(out / "tokenizer.json")
    elif (out / "tokenizer.json").exists():
        bpe = BPE.load(out / "tokenizer.json")
    else:
        print("tokenizer eğitiliyor...", flush=True)
        bpe = train_tokenizer(out, a.vocab, a.seed, extra_texts=[r["src"] for r in ext_rows])
    print(f"tokenizer: {bpe.vocab_size} parça", flush=True)

    cfg = GptConfig(vocab_size=bpe.vocab_size + N_PTR, ctx=a.ctx, d_model=a.d_model, n_layer=a.n_layer,
                    n_head=a.n_head, d_ff=a.d_ff)
    model = TinyGpt(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.05)
    step = 0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = TinyGpt(GptConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.05)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"devam: adım {step}", flush=True)
    print(f"model: {model.n_params():,} parametre", flush=True)

    teacher = None
    if a.teacher:
        tk = torch.load(a.teacher, map_location="cpu", weights_only=False)
        teacher = TinyGpt(GptConfig(**tk["config"]))
        teacher.load_state_dict(tk["model"])
        teacher.eval()
        assert teacher.config.vocab_size == cfg.vocab_size, "öğretmen ve öğrenci aynı tokenizer'ı kullanmalı"
        print(f"damıtma: öğretmen {teacher.n_params():,} parametre, alpha={a.kd_alpha}, T={a.kd_temp}", flush=True)

    distilled = load_distilled(a.distilled)
    if distilled:
        print(f"damıtılmış örnek: {len(distilled)}", flush=True)
    seed_rows = load_distilled([str(SEED_SET)]) if SEED_SET.exists() else []
    from .massive import EXT_FRAC
    stream = Stream(bpe, cfg.ctx, a.seed + step, distilled, a.distill_frac, seed_rows,
                    ext_rows=ext_rows, ext_frac=EXT_FRAC if a.ext_frac is None else a.ext_frac)

    t0 = time.time()
    budget = a.time_budget_min * 60
    log = open(out / "train.log", "a", encoding="utf-8")

    def save(final: bool = False) -> None:
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "final": final}, ckpt_path)

    model.train()
    ema = None
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        x, y = stream.batch(a.batch)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        if teacher is not None:
            with torch.no_grad():
                tl = teacher(x)
            mask = (y != -100).unsqueeze(-1)
            T = a.kd_temp
            kl = F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(tl / T, -1), log_target=True,
                          reduction="none")
            kd = (kl * mask).sum() / mask.sum().clamp(min=1) * T * T
            loss = (1 - a.kd_alpha) * loss + a.kd_alpha * kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            msg = f"adım {step} kayıp {ema:.4f} süre {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if step % a.ckpt_every == 0:
            save()
        if time.time() - t0 > budget:
            print("süre bütçesi doldu; checkpoint kaydedildi, sonraki iş devam edecek", flush=True)
            break
    save(final=step >= a.steps)
    (out / "status.json").write_text(json.dumps({"step": step, "target_steps": a.steps,
                                                 "done": step >= a.steps, "loss": ema,
                                                 "ext_rows": len(ext_rows), "ext_frac": stream.ext_frac}), encoding="utf-8")
    print(f"bitti: adım {step}/{a.steps}", flush=True)


if __name__ == "__main__":
    main()
