"""Sohbet beyni: Orion'un sıfırdan eğitilen Türkçe dil modeli.

Adımlar (hepsi CI'da, gelişim deposunda):
  prepare  → Türkçe Vikipedi metni (açık lisans, CC BY-SA) + Orion'un kendi kişilik diyalogları;
             tokenizer (BPE, 8192 parça) eğitilir, metin belirteçlere çevrilip uint16 dosyaya yazılır
  train    → rastgele pencerelerle dil modeli eğitimi; süre bütçeli, kaldığı yerden devam eder
  export   → ORN1 (int8) + parity + örnek cevaplar + manifest (doğrulama kaybı, kişilik testi)

Biçim: <bos>Kullanıcı: ...\\nOrion: ...<eos>  (düz metin belgeleri <bos>metin<eos>)
Çıkarım istemi "Kullanıcı: ...\\nOrion:" ile biter (sonda BOŞLUK YOK): eğitimde cevap boşluklu bir belirteçle
(" Merhaba") başlar; istem "Orion: " ile biterse BPE tek başına bir " " belirteci üretir, model bunu cevaptan önce
hiç görmediği için sözcüğün ortasından başlar (70000 adımlık modelde sınav 0.3 → 0.7).
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
USER, BOT = "Kullanıcı: ", "Orion: "  # BOT: EĞİTİM biçimi (dialogs.format_dialog); değişmez
PROMPT_BOT = BOT.rstrip()              # çıkarım istemi sonu: "Orion:" (sonda boşluk yok)

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
from .dialogs import QA_TESTS, open_dialogues, persona_dialogues, qa_judge, qa_score  # noqa: E402
from . import dialogs as _dialogs  # noqa: E402

EXTRA_LICENSES = {"apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0"}


def extra_dialogues(src: str, cache: Path) -> tuple[list[str], dict]:
    """Ek diyalog dosyası (yol ya da URL; .jsonl ya da .jsonl.gz; translate_data çıktısı biçimi).
    Yalnız training_allowed=true VE izinli lisanslı satırlar alınır; her tur dialogs.clean_pair'den geçer
    (sınav soruları, başka model kimliği, Türkçe olmayan cevap atılır). Çok turlu satırlar ("turns") turlar
    alt alta eğitim biçimiyle birleştirilir; ilk tur geçemezse satır düşer, sonraki tur geçemezse orada kesilir."""
    import gzip
    from collections import Counter
    path = Path(src)
    if src.startswith(("http://", "https://")):
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / ("extra-dialogs.jsonl.gz" if src.split("?")[0].endswith(".gz") else "extra-dialogs.jsonl")
        if not path.exists():
            print(f"ek diyalog indiriliyor: {src}", flush=True)
            urllib.request.urlretrieve(src, path)
    op = gzip.open if str(path).endswith(".gz") else open
    texts: list[str] = []
    drop: Counter = Counter()
    by_src: Counter = Counter()
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("training_allowed") is not True:
                drop["egitim_izni_yok"] += 1
                continue
            if str(r.get("license", "")).lower() not in EXTRA_LICENSES:
                drop["lisans"] += 1
                continue
            turns = r.get("turns") or [[r.get("q", ""), r.get("a", "")]]
            parts = []
            for q, a in turns:
                c = _dialogs.clean_pair(q, a)
                if c is None:
                    break
                parts.append(_dialogs.format_dialog(*c))
            if not parts:
                drop["suzgec_ya_da_sinav"] += 1
                continue
            texts.append("\n".join(parts))
            by_src[r.get("source", "?")] += 1
    return texts, {"ceviri_cift": len(texts), "kaynaklar": dict(by_src), "atilan": dict(drop), "dosya": src}


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
            tokenizer: Path | None = None, open_data: bool = True, extra_dialogs: str | None = None) -> dict:
    """tokenizer: önceki eğitimin tokenizer'ı (Kaggle'da kaldığı yerden devam ederken AYNI olmalı).
    extra_dialogs: ek (ör. çeviri) diyalog jsonl yolu/URL'si; bkz. extra_dialogues. Yoksa davranış değişmez."""
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
    extra_texts, extra_stats = extra_dialogues(extra_dialogs, out / "extra") if extra_dialogs else ([], None)
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
        "dialog_tokens": encode_to(out / "dialog.bin", bpe, dialogs + open_texts + open_texts + extra_texts),
        "persona_dialogs": len(dialogs), "open_dialogs": open_stats,
        "vocab": bpe.vocab_size,
        "source": "Türkçe Vikipedi (wikimedia/wikipedia 20231101.tr, CC BY-SA 3.0/GFDL) + Orion kişilik/yetenek diyalogları + "
                  "CohereForAI/aya_dataset (Apache-2.0) + OpenAssistant/oasst2 (Apache-2.0) Türkçe diyalogları",
    }
    if extra_stats is not None:
        stats["ceviri_cift"] = extra_stats["ceviri_cift"]
        stats["ceviri"] = extra_stats
        stats["source"] += (" + İngilizce açık lisanslı diyalogların makine çevirisi (" +
                            ", ".join(sorted(extra_stats["kaynaklar"])) + "; ayrıntı: ceviri)")
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
          threads: int = 0, config: dict | None = None, resume_from: Path | None = None, grad_accum: int = 1,
          save_every: int = 500, exam_every: int = 0) -> dict:
    """save_every: ckpt.pt + status.json yazma aralığı (erken durdurma için ara durum).
    exam_every > 0: o aralıkla sınav (run_exam, 40 soru, açgözlü çözme) yapılır; exam.jsonl'e yazılır ve en iyi
    qa_score'lu ağırlıklar best.pt'ye (ckpt.pt biçiminde, export'a verilebilir) kaydedilir. 0 = kapalı (varsayılan)."""
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

    def save(path: Path = ckpt, with_opt: bool = True):
        tmp = path.with_suffix(".tmp")
        ck = {"config": cfg.to_dict(), "model": model.state_dict(), "step": step}
        if with_opt:
            ck["opt"] = opt.state_dict()
        torch.save(ck, tmp)
        tmp.replace(path)

    exams: list[dict] = []
    best_prev = out / "exam.jsonl"
    if best_prev.exists():  # aynı oturumda kaldığı yerden: önceki sınavlar
        exams = [json.loads(x) for x in best_prev.read_text(encoding="utf-8").splitlines() if x.strip()]

    def write_status(final: bool = False, vl: float | None = None) -> dict:
        st = {"step": step, "target_steps": steps, "done": step >= steps, "train_loss": ema,
              "tokens_seen": step * batch * grad_accum * cfg.ctx, "params": model.n_params(), "device": str(dev),
              "final": final}
        if vl is not None:
            st.update(val_loss=vl, val_ppl=math.exp(vl) if vl == vl else None)
        if exams:
            best = max(exams, key=lambda e: e["qa_score"])  # ilk en iyi = best.pt'deki
            st.update(last_exam=exams[-1], best_exam=best)
        (out / "status.json").write_text(json.dumps(st), encoding="utf-8")
        return st

    def exam():
        ex = run_exam(model, bpe)
        model.train()
        e = {"step": step, "qa_score": ex["qa_score"], "persona_ok": ex["persona_ok"]}
        is_best = not exams or e["qa_score"] > max(x["qa_score"] for x in exams)
        exams.append(e)
        with open(out / "exam.jsonl", "a", encoding="utf-8") as fx:
            fx.write(json.dumps(e) + "\n")
        if is_best:
            save(out / "best.pt", with_opt=False)
        print(f"sınav adım {step}: {e['qa_score']:.3f}{' (en iyi)' if is_best else ''}", flush=True)

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
        if exam_every and step % exam_every == 0:
            exam()
        if step % save_every == 0:
            save()
            write_status()
        if time.time() - t0 > budget_min * 60:
            print("süre bütçesi doldu; sonraki iş kaldığı yerden devam edecek", flush=True)
            break
    save()
    vl = val_loss(model, data, cfg.ctx)
    status = write_status(final=True, vl=vl)
    print(json.dumps(status), flush=True)
    return status


# --------------------------------------------------------------------------- telefona aktarma

PERSONA_TESTS = [("Sen kimsin?", "Orion"), ("Merhaba", None), ("Çok sıkıldım", None)]
# QA_TESTS (dialogs.py): eğitimde geçmeyen soru biçimleri; "doğru soruya doğru cevap" sınavı


def prompt_for(question: str) -> str:
    """Çıkarım istemi; "Orion:" ile biter (sonda boşluk YOK, bkz. modül açıklaması)."""
    return f"{USER}{question}\n{PROMPT_BOT}"


def clean_reply(raw: str) -> str:
    """Cevabı temizler (ChatBrain.cleanReply ile aynı): baştaki "Orion:" atılır; "Kullanıcı:" ile başlıyorsa
    sonraki "Orion:" cevabı alınır; sonraki tura geçmeden kesilir."""
    u, b = USER.strip(), BOT.strip()
    s = raw.strip()
    if s.startswith(u):
        s = s.split(f"\n{b}", 1)[1] if f"\n{b}" in s else ""
    s = s.strip()
    if s.startswith(b):
        s = s[len(b):]
    return s.split(f"\n{u}")[0].split(f"\n{b}")[0].strip()


@torch.no_grad()
def generate(model: TinyGpt, bpe: BPE, text: str, max_new: int = 80, temperature: float = 0.0, top_k: int = 40,
             seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = [BOS] + bpe.encode(text)
    out: list[int] = []
    model.eval()
    for _ in range(max_new):
        x = torch.tensor([ids[-model.config.ctx:]], device=next(model.parameters()).device)
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
        t = s.strip()
        # Model yeni bir tur uydurduysa ("Kullanıcı: ...\nOrion: ...") cevabın ardındaki ilk turda dur
        body = t.split(f"\n{BOT.strip()}", 1)[1] if t.startswith(USER.strip()) and f"\n{BOT.strip()}" in t else t
        if not t.startswith(USER.strip()) or body is not t:
            if f"\n{USER.strip()}" in body:
                return clean_reply(s)
    return clean_reply(bpe.decode(out))


def run_exam(model: TinyGpt, bpe: BPE, max_new: int = 50) -> dict:
    """Sınav (QA_TESTS) + kişilik testi; açgözlü (greedy) çözme, export_chat ile aynı ayarlar.
    Dönen: {"qa_score", "persona_ok", "persona_text", "answers": [{soru, tur, cevap, dogru, neden}]}"""
    texts = [generate(model, bpe, prompt_for(q), max_new=max_new) for q, *_ in QA_TESTS]
    answers = [dict(r, cevap=a) for r, a in zip(qa_judge(texts), texts)]
    persona_text = generate(model, bpe, prompt_for(PERSONA_TESTS[0][0]), max_new=40)
    return {"qa_score": qa_score(texts), "persona_ok": PERSONA_TESTS[0][1] in persona_text,
            "persona_text": persona_text, "answers": answers}


def evaluate_release(dir: Path) -> dict:
    """İndirilmiş bir sohbet modelini (dir/model.bin + dir/tokenizer.json) yerel olarak sınar; dosya yazmaz.
    Dönen: {"qa_score": float, "persona_ok": bool, "persona_text": str, "answers": [...]} (bkz. run_exam)."""
    from .export import load_quantized
    dir = Path(dir)
    model = load_quantized(dir / "model.bin")
    bpe = BPE.load(dir / "tokenizer.json")
    return run_exam(model, bpe)


def export_chat(run: Path, data: Path, out: Path, ckpt_name: str = "ckpt.pt", cog_exam: bool = True) -> dict:
    """ckpt_name: "best.pt" ile eğitim içi sınavın en iyi ağırlıkları dışa aktarılır (erken durdurma).
    cog_exam: Bilişsel Yetenek Sınavı (cogexam) da yapılır; bölüm puanları manifest'e "cog_bilissel"/"cog_ilkesel"
    olarak YALNIZCA RAPOR için yazılır (chat-model-floor.json'da kapısı yok), cevaplar cogexam.json'a."""
    from .export import export, load_quantized
    out.mkdir(parents=True, exist_ok=True)
    info = export(run / ckpt_name, out / "model.bin")
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
    if ckpt_name != "ckpt.pt":  # adım, dışa aktarılan checkpoint'ten
        ck_step = torch.load(run / ckpt_name, map_location="cpu", weights_only=False).get("step")
        status = dict(status, step=ck_step, tokens_seen=None)
    vl = val_loss(m, data, m.config.ctx) if (data / "val.bin").exists() else float("nan")
    persona_ok = "Orion" in cases[0]["greedy_text"]
    ex = run_exam(m, bpe)
    qa = ex["answers"]
    (out / "samples.json").write_text(json.dumps(samples + [{"sinav": x} for x in qa], ensure_ascii=False, indent=1), encoding="utf-8")
    qa_ok = ex["qa_score"]
    manifest = {"kind": "chat", "bytes": info["bytes"], "sha256": info["sha256"], "config": m.config.to_dict(),
                "params": m.n_params(), "step": status.get("step"), "tokens_seen": status.get("tokens_seen"),
                "val_loss_int8": vl, "val_ppl_int8": math.exp(vl) if vl == vl else None, "persona_ok": persona_ok, "qa_score": qa_ok,
                "data": json.loads((data / "data.json").read_text()) if (data / "data.json").exists() else {}}
    if cog_exam:
        from . import cogexam
        if cogexam.EXAM_PATH.exists():
            cog = cogexam.run(m, bpe)
            (out / "cogexam.json").write_text(json.dumps(cog, ensure_ascii=False, indent=1), encoding="utf-8")
            manifest["cog_bilissel"] = cog["bilissel"]["score"]  # ayrı puanlar; ortalaması alınmaz
            manifest["cog_ilkesel"] = cog["ilkesel"]["score"]
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
    p.add_argument("--extra-dialogs", help="ek diyalog jsonl(.gz) yolu ya da URL'si (ör. ceviri-* Release'i); "
                                           "yalnız training_allowed=true satırlar")
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
    t.add_argument("--save-every", type=int, default=500, help="ckpt.pt + status.json aralığı (adım)")
    t.add_argument("--exam-every", type=int, default=0, help=">0: bu aralıkla sınav; en iyisi best.pt (0 = kapalı)")
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--ckpt", default="ckpt.pt", help="run klasöründeki checkpoint (best.pt: en iyi sınav)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(Path(a.out), a.max_chars, a.vocab, tokenizer=Path(a.tokenizer) if a.tokenizer else None,
                                 open_data=not a.no_open_data, extra_dialogs=a.extra_dialogs), ensure_ascii=False, indent=1))
    elif a.cmd == "train":
        train(Path(a.data), Path(a.out), a.steps, a.budget_min, a.batch, a.lr, a.dialog_frac, a.threads,
              config=SIZES[a.size], resume_from=Path(a.resume_from) if a.resume_from else None, grad_accum=a.grad_accum,
              save_every=a.save_every, exam_every=a.exam_every)
    else:
        export_chat(Path(a.run), Path(a.data), Path(a.out), a.ckpt)


if __name__ == "__main__":
    main()
