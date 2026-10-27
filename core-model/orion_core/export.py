"""Modeli telefonda çalışacak biçime aktarır: ORN1 ikili dosyası (int8 ağırlık + satır ölçeği).

Neden ONNX/GGUF/TFLite değil? Model 1-5M parametre; saf Kotlin ile çalıştırmak yerel
kütüphane (NDK) gerektirmez, APK'yı küçük tutar ve JVM'de birebir test edilebilir.

Biçim (küçük-endian):
  "ORN1" | uint32 başlık_uzunluğu | başlık JSON (utf-8) | 4 bayta hizalı veri
  başlık: {"config": {...}, "tensors": [{"name","shape","dtype":"i8"|"f32","offset","scale_offset"}]}
  i8 tensörler: her satır için float32 ölçek (scale_offset) + int8 değerler (offset)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

from .bpe import BOS, EOS, BPE
from .copy import N_PTR, decode_output
from .model import GptConfig, TinyGpt
from .train import prompt_ids

MAGIC = b"ORN1"


def tensor_order(cfg: GptConfig) -> list[str]:
    names = ["tok.weight", "pos.weight"]
    for i in range(cfg.n_layer):
        for part in ("ln1.weight", "ln1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                     "ln2.weight", "ln2.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
            names.append(f"blocks.{i}.{part}")
    return names + ["ln_f.weight", "ln_f.bias"]


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def export(ckpt: Path, out: Path) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = GptConfig(**ck["config"])
    sd = ck["model"]
    header_tensors = []
    blobs: list[bytes] = []
    offset = 0
    deq_state = {}
    for name in tensor_order(cfg):
        t = sd[name].float()
        if t.dim() == 2:
            q, s = quantize_rows(t)
            sb = s.numpy().astype("<f4").tobytes()
            qb = q.numpy().tobytes()
            qb += b"\0" * ((-len(qb)) % 4)
            header_tensors.append({"name": name, "shape": list(t.shape), "dtype": "i8",
                                   "scale_offset": offset, "offset": offset + len(sb)})
            blobs += [sb, qb]
            offset += len(sb) + len(qb)
            deq_state[name] = q.float() * s[:, None]
        else:
            b = t.numpy().astype("<f4").tobytes()
            header_tensors.append({"name": name, "shape": list(t.shape), "dtype": "f32", "offset": offset})
            blobs.append(b)
            offset += len(b)
            deq_state[name] = t
    header = json.dumps({"config": cfg.to_dict(), "tensors": header_tensors}, separators=(",", ":")).encode()
    pad = (-(8 + len(header))) % 4
    header += b" " * pad
    data = MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs)
    out.write_bytes(data)
    return {"config": cfg, "deq_state": deq_state, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load_quantized(model_bin: Path) -> TinyGpt:
    """ORN1 dosyasını geri yükler (değerlendirme, telefondakiyle AYNI ağırlıklarla yapılır)."""
    raw = model_bin.read_bytes()
    assert raw[:4] == MAGIC
    (hl,) = struct.unpack("<I", raw[4:8])
    head = json.loads(raw[8:8 + hl])
    base = 8 + hl
    cfg = GptConfig(**head["config"])
    m = TinyGpt(cfg)
    sd = {}
    for t in head["tensors"]:
        shape = t["shape"]
        if t["dtype"] == "i8":
            rows = shape[0]
            s = torch.frombuffer(bytearray(raw[base + t["scale_offset"]: base + t["scale_offset"] + 4 * rows]), dtype=torch.float32)
            n = shape[0] * shape[1]
            q = torch.frombuffer(bytearray(raw[base + t["offset"]: base + t["offset"] + n]), dtype=torch.int8).float()
            sd[t["name"]] = q.view(*shape) * s[:, None]
        else:
            n = 1
            for d in shape:
                n *= d
            sd[t["name"]] = torch.frombuffer(bytearray(raw[base + t["offset"]: base + t["offset"] + 4 * n]),
                                             dtype=torch.float32).view(*shape).clone()
    m.load_state_dict(sd)
    m.eval()
    return m


PARITY_PROMPTS = [
    ("t2c", "yarın saat 10'da toplantı hatırlat"),
    ("t2c", "not al: süt ve ekmek al"),
    ("t2c", "rakip fiyatları hakkında araştırma yap"),
    ("plan", '{"args":{"text":"x"},"cmd":"note.write"}'),
    # Ek bölme (Java regex = Python regex olmalı) ve mastar kopyalama
    ("t2c", "pazartesi 9'da faturayı ödemeyi hatırlat"),
    ("t2c", "telefonda notlarımı ve hatırlatıcılarımı göster"),
    ("ans", '{"err_code":null,"id":"g","next":[],"result":{"kind":"reminder","text":"annemi aramak"},"status":"ok","v":1}'),
]


def parity_vectors(model: TinyGpt, bpe: BPE) -> dict:
    """Kotlin çalıştırıcısının birebir aynı sonucu verdiğini sınamak için örnekler."""
    out = []
    for task, src in PARITY_PROMPTS:
        ids = prompt_ids(bpe, task, src)
        with torch.no_grad():
            logits = model(torch.tensor([ids]))[0, -1]
        gen = model.greedy(ids, EOS, max_new=64)
        out.append({"task": task, "src": src, "prompt_ids": ids, "last_logits_head": [round(float(x), 5) for x in logits[:16]],
                    "argmax": int(torch.argmax(logits)), "greedy_ids": gen, "greedy_text": decode_output(bpe, ids[2:-1], gen)})
    return {"bos": BOS, "eos": EOS, "n_ptr": N_PTR, "cases": out}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="eğitim klasörü (ckpt.pt + tokenizer.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = export(run / "ckpt.pt", out / "model.bin")
    bpe = BPE.load(run / "tokenizer.json")
    bpe.save(out / "tokenizer.json")
    m = load_quantized(out / "model.bin")
    (out / "parity.json").write_text(json.dumps(parity_vectors(m, bpe), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"model.bin {info['bytes'] / 1e6:.2f} MB sha256={info['sha256'][:16]}")


if __name__ == "__main__":
    main()
