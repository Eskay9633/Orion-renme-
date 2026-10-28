"""Sohbet beyni: Orion'un sıfırdan eğitilen Türkçe dil modeli.

Adımlar (hepsi CI'da, gelişim deposunda):
  prepare  → Türkçe Vikipedi metni (açık lisans, CC BY-SA) + Orion'un kendi kişilik diyalogları;
             tokenizer (BPE, 8192 parça) eğitilir, metin belirteçlere çevrilip uint16 dosyaya yazılır
  train    → rastgele pencerelerle dil modeli eğitimi; süre bütçeli, kaldığı yerden devam eder
  export   → ORN1 (int8) + parity + örnek cevaplar + manifest (doğrulama kaybı, kişilik testi)

Biçim: <bos>Kullanıcı: ...\\nOrion: ...<eos>  (düz metin belgeleri <bos>metin<eos>)
Telefonda: orion-core/.../model/ChatBrain.kt aynı biçimi kurar.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .bpe import BOS, EOS, BPE
from .model import GptConfig, TinyGpt

HF_API = "https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/main/20231101.tr"
HF_FILE = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
CUT_SECTIONS = ("\nKaynakça", "\nKaynaklar", "\nDış bağlantılar", "\nAyrıca bakınız", "\nNotlar")
USER, BOT = "Kullanıcı: ", "Orion: "

CHAT_CONFIG = dict(ctx=256, d_model=384, n_layer=8, n_head=6, d_ff=1536)
# Boyutlar: küçük GitHub işlemcisinde (~17M), orta/büyük ekran kartında (Kaggle) eğitilir.
SIZES = {
    "kucuk": CHAT_CONFIG,                                                        # ~17M
    "orta": dict(ctx=512, d_model=512, n_layer=12, n_head=8, d_ff=2048),         # ~42M
    "buyuk": dict(ctx=512, d_model=768, n_layer=12, n_head=12, d_ff=3072),       # ~92M
}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- veri

def clean_article(text: str) -> str:
    for c in CUT_SECTIONS:
        i = text.find(c)
        if i > 0:
            text = text[:i]
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def wikipedia_texts(cache: Path, max_chars: int):
    """Hugging Face'teki wikimedia/wikipedia 20231101.tr parquet dosyalarından makale metni (doğrulanmadı: adres CI'da denenir)."""
    import pyarrow.parquet as pq
    cache.mkdir(parents=True, exist_ok=True)
    files = json.loads(urllib.request.urlopen(HF_API, timeout=60).read())
    paths = sorted(f["path"] for f in files if f["path"].endswith(".parquet"))
    total = 0
    for p in paths:
        local = cache / Path(p).name
        if not local.exists():
            print(f"indiriliyor: {p}", flush=True)
            urllib.request.urlretrieve(HF_FILE + p, local)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=2000, columns=["title", "text"]):
            for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
                t = clean_article(text or "")
                if len(t) < 300:
                    continue
                yield t
                total += len(t)
                if total >= max_chars:
                    return


# Kişilik/yetenek diyalogları ve açık lisanslı Türkçe diyaloglar: dialogs.py
from .dialogs import QA_TESTS, open_dialogues, persona_dialogues, qa_score  # noqa: E402


def encode_to(path: Path, bpe: BPE, texts) -> int:
    n = 0
    with open(path, "wb") as f:
        buf: list[int] = []
        for t in texts:
            buf.append(BOS)
            buf.extend(bpe.encode(t))
            buf.append(EOS)
            if len(buf) > 1_000_000:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                n += len(buf)
                buf = []
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            n += len(buf)
    return n


def prepare(out: Path, max_chars: int, vocab: int, tok_sample_chars: int = 30_000_000, texts=None,
            tokenizer: Path | None = None, open_data: bool = True) -> dict:
    """tokenizer: önceki eğitimin tokenizer'ı (Kaggle'da kaldığı yerden devam ederken AYNI olmalı)."""
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.txt"
    if not raw.exists():
        it = texts if texts is not None else wikipedia_texts(out / "hf", max_chars)
        with open(raw, "w", encoding="utf-8") as f:
            for t in it:
                f.write(t.replace("\x00", " ") + "\x00")
    docs = raw.read_text(encoding="utf-8").split("\x00")
    docs = [d for d in docs if d.strip()]
    dialogs = persona_dialogues(60000)
    open_texts, open_stats = open_dialogues(out / "hf-dialog") if open_data else ([], {})
    if tokenizer is not None and Path(tokenizer).exists() and not (out / "tokenizer.json").exists():
        import shutil
        shutil.copy(tokenizer, out / "tokenizer.json")
    if not (out / "tokenizer.json").exists():
        sample, size = [], 0
        for d in docs:
            sample.append(d)
            size += len(d)
            if size > tok_sample_chars:
                break
        bpe = BPE.train(sample + dialogs[:2000], vocab_size=vocab)
        bpe.save(out / "tokenizer.json")
    bpe = BPE.load(out / "tokenizer.json")
    bpe._cache_limit = 2_000_000
    rng = random.Random(1)
    rng.shuffle(docs)
    n_val = max(1, len(docs) // 200)
    stats = {
        "docs": len(docs), "chars": sum(map(len, docs)),
        "train_tokens": encode_to(out / "train.bin", bpe, docs[n_val:]),
        "val_tokens": encode_to(out / "val.bin", bpe, docs[:n_val]),
        "dialog_tokens": encode_to(out / "dialog.bin", bpe, dialogs + open_texts + open_texts),
        "persona_dialogs": len(dialogs), "open_dialogs": open_stats,
        "vocab": bpe.vocab_size,
        "source": "Türkçe Vikipedi (wikimedia/wikipedia 20231101.tr, CC BY-SA 3.0/GFDL) + Orion kişilik/yetenek diyalogları + "
                  "CohereForAI/aya_dataset (Apache-2.0) + OpenAssistant/oasst2 (Apache-2.0) Türkçe diyalogları",
    }
    (out / "data.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


# --------------------------------------------------------------------------- eğitim

class Windows:
    def __init__(self, data: Path, ctx: int, seed: int, dialog_frac: float):
        self.train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
        self.dialog = np.memmap(data / "dialog.bin", dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)
        self.dialog_frac = dialog_frac

    def batch(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        for _ in range(n):
            src = self.dialog if self.rng.random() < self.dialog_frac else self.train
            i = int(self.rng.integers(0, len(src) - self.ctx - 1))
            xs.append(np.asarray(src[i:i + self.ctx + 1], dtype=np.int64))
        t = torch.from_numpy(np.stack(xs))
        return t[:, :-1], t[:, 1:]


@torch.no_grad()
def val_loss(model: TinyGpt, data: Path, ctx: int, n: int = 40) -> float:
    dev = next(model.parameters()).device
    v = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    if len(v) <= ctx + 1:
        return float("nan")
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(n):
        i = int(rng.integers(0, len(v) - ctx - 1))
        t = torch.from_numpy(np.asarray(v[i:i + ctx + 1], dtype=np.int64))[None].to(dev)
        logits = model(t[:, :-1]).float()
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1)).item())
    model.train()
    return float(sum(losses) / len(losses))


def lr_at(step: int, total: int, base: float, warmup: int = 500) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(data: Path, out: Path, steps: int, budget_min: float, batch: int, lr: float, dialog_frac: float,
          threads: int = 0, config: dict | None = None, resume_from: Path | None = None, grad_accum: int = 1) -> dict:
    if threads:
        torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)
    bpe = BPE.load(data / "tokenizer.json")
    bpe.save(out / "tokenizer.json")
    ckpt = out / "ckpt.pt"
    if not ckpt.exists() and resume_from is not None and Path(resume_from).exists():
        import shutil
        shutil.copy(resume_from, ckpt)  # Kaggle: önceki oturumun çıktısından devam
    dev = device()
    cfg = GptConfig(vocab_size=bpe.vocab_size, **(config or CHAT_CONFIG))
    model = TinyGpt(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    step = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TinyGpt(GptConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        cfg = model.config
        print(f"devam: adım {step}", flush=True)
    model.to(dev)
    # Optimizer durumu da aynı cihaza taşınsın
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"sohbet modeli: {model.n_params():,} parametre, cihaz {dev}", flush=True)
    win = Windows(data, cfg.ctx, 1000 + step, dialog_frac)
    t0 = time.time()
    log = open(out / "train.log", "a", encoding="utf-8")

    def save():
        tmp = out / "ckpt.tmp"
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
        tmp.replace(ckpt)

    model.train()
    ema = None
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x, y = win.batch(batch)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)) / grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        loss = loss * grad_accum
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            msg = f"adım {step} kayıp {ema:.3f} süre {time.time() - t0:.0f}s"
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
        if step % 500 == 0:
            save()
        if time.time() - t0 > budget_min * 60:
            print("süre bütçesi doldu; sonraki iş kaldığı yerden devam edecek", flush=True)
            break
    save()
    vl = val_loss(model, data, cfg.ctx)
    status = {"step": step, "target_steps": steps, "done": step >= steps, "train_loss": ema,
              "val_loss": vl, "val_ppl": math.exp(vl) if vl == vl else None,
              "tokens_seen": step * batch * grad_accum * cfg.ctx, "params": model.n_params(), "device": str(dev)}
    (out / "status.json").write_text(json.dumps(status), encoding="utf-8")
    print(json.dumps(status), flush=True)
    return status


# --------------------------------------------------------------------------- telefona aktarma

PERSONA_TESTS = [("Sen kimsin?", "Orion"), ("Merhaba", None), ("Çok sıkıldım", None)]
# QA_TESTS (dialogs.py): eğitimde geçmeyen soru biçimleri; "doğru soruya doğru cevap" sınavı


def prompt_for(question: str) -> str:
    return f"{USER}{question}\n{BOT}"


@torch.no_grad()
def generate(model: TinyGpt, bpe: BPE, text: str, max_new: int = 80, temperature: float = 0.0, top_k: int = 40,
             seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = [BOS] + bpe.encode(text)
    out: list[int] = []
    model.eval()
    for _ in range(max_new):
        x = torch.tensor([ids[-model.config.ctx:]])
        logits = model(x)[0, -1]
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            v, i = torch.topk(logits / temperature, top_k)
            nxt = int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=g)])
        if nxt == EOS:
            break
        ids.append(nxt)
        out.append(nxt)
        s = bpe.decode(out)
        if f"\n{USER.strip()}" in s:
            return s.split(f"\n{USER.strip()}")[0].strip()
    return bpe.decode(out).strip()


def export_chat(run: Path, data: Path, out: Path) -> dict:
    from .export import export, load_quantized
    out.mkdir(parents=True, exist_ok=True)
    info = export(run / "ckpt.pt", out / "model.bin")
    bpe = BPE.load(run / "tokenizer.json")
    bpe.save(out / "tokenizer.json")
    m = load_quantized(out / "model.bin")
    cases = []
    for q, _ in PERSONA_TESTS:
        p = prompt_for(q)
        ids = [BOS] + bpe.encode(p)
        with torch.no_grad():
            logits = m(torch.tensor([ids]))[0, -1]
        cases.append({"prompt": p, "prompt_ids": ids, "last_logits_head": [round(float(x), 5) for x in logits[:16]],
                      "greedy_text": generate(m, bpe, p, max_new=40)})
    (out / "parity.json").write_text(json.dumps({"bos": BOS, "eos": EOS, "cases": cases}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    samples = []
    for q in ["Sen kimsin?", "Merhaba, nasılsın?", "Çok sıkıldım, ne yapayım?", "İstanbul hakkında bir şey anlat.",
              "Kodlama öğrenmek istiyorum.", "Bana bir hobi öner."]:
        samples.append({"soru": q, "cevap": generate(m, bpe, prompt_for(q), temperature=0.7, seed=1)})
    for doc in ["Türkiye'nin başkenti", "Fotosentez", "Osmanlı İmparatorluğu"]:
        samples.append({"metin": doc, "devam": generate(m, bpe, doc, temperature=0.7, seed=2)})
    (out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    vl = val_loss(m, data, m.config.ctx) if (data / "val.bin").exists() else float("nan")
    persona_ok = "Orion" in cases[0]["greedy_text"]
    qa = [{"soru": q, "cevap": generate(m, bpe, prompt_for(q), max_new=50)} for q, _ in QA_TESTS]
    (out / "samples.json").write_text(json.dumps(samples + [{"sinav": x} for x in qa], ensure_ascii=False, indent=1), encoding="utf-8")
    qa_ok = qa_score([x["cevap"] for x in qa])
    manifest = {"kind": "chat", "bytes": info["bytes"], "sha256": info["sha256"], "config": m.config.to_dict(),
                "params": m.n_params(), "step": status.get("step"), "tokens_seen": status.get("tokens_seen"),
                "val_loss_int8": vl, "val_ppl_int8": math.exp(vl) if vl == vl else None, "persona_ok": persona_ok, "qa_score": qa_ok,
                "data": json.loads((data / "data.json").read_text()) if (data / "data.json").exists() else {}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(samples, ensure_ascii=False, indent=1))
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--max-chars", type=int, default=600_000_000)
    p.add_argument("--vocab", type=int, default=8192)
    p.add_argument("--tokenizer", help="önceki eğitimin tokenizer.json'ı (devam ederken aynı tokenizer şart)")
    p.add_argument("--no-open-data", action="store_true", help="açık lisanslı diyalog verisini indirme (test/çevrimdışı)")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=120_000)
    t.add_argument("--budget-min", type=float, default=330)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=6e-4)
    t.add_argument("--dialog-frac", type=float, default=0.2)
    t.add_argument("--threads", type=int, default=0)
    t.add_argument("--size", choices=list(SIZES), default="kucuk")
    t.add_argument("--resume-from", help="önceki checkpoint (Kaggle: önceki oturumun çıktısı)")
    t.add_argument("--grad-accum", type=int, default=1)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(Path(a.out), a.max_chars, a.vocab, tokenizer=Path(a.tokenizer) if a.tokenizer else None,
                                 open_data=not a.no_open_data), ensure_ascii=False, indent=1))
    elif a.cmd == "train":
        train(Path(a.data), Path(a.out), a.steps, a.budget_min, a.batch, a.lr, a.dialog_frac, a.threads,
              config=SIZES[a.size], resume_from=Path(a.resume_from) if a.resume_from else None, grad_accum=a.grad_accum)
    else:
        export_chat(Path(a.run), Path(a.data), Path(a.out))


if __name__ == "__main__":
    main()
