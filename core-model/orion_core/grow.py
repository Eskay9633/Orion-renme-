"""Genişlik büyütme: eğitilmiş bir sohbet beynini (ör. orta, d_model 512) daha geniş bir yapıya (ör. büyük, 768)
İŞLEVİ KORUYARAK taşır; büyük model ilk adımda kaynakla (neredeyse) aynı çıktıyı verir, sonra eğitimle açılır.

Yöntem ("ortalama izleyen sıfır varyanslı genişleme"; Net2Net/bert2BERT ailesinden, LayerNorm için tam):
  * Eski ağırlıklar yeni matrislerin sol-üst bloğuna aynen kopyalanır. Başlık boyutu (64) değişmez; 8 → 12 başlıkta
    eski başlıklar 0..7, yeni başlıklar 8..11'dir (q/k/v satırları başlık başına bitişik).
  * Kalıntı akışının (residual) YENİ her boyutu, eski boyutların ORTALAMASINI taşır: gömme (tok/pos) yeni sütunları
    eski sütunların satır ortalaması; proj/fc2'nin yeni çıkış satırları eski satırların ortalaması (ağırlık+sapma).
    Toplama doğrusal olduğundan bu eşitlik her katmanda kendiliğinden korunur.
  * LayerNorm uyarısı: LN ortalama/varyansı TÜM boyutlar üzerinden alır. Yeni boyutlar = ortalama olduğundan
    ortalama değişmez, varyans ise d_eski/d_yeni oranında küçülür (512/768 = 2/3). Yalnızca γ'yı büyütmek eps
    (1e-5) yüzünden tam değildir (ilk katmanlarda σ² ~1e-4: 86000 modelinde logit farkı ~4e-2 çıktı). Bu yüzden
    kalıntı akışının TAMAMI s = sqrt(d_yeni/d_eski) ile ölçeklenir (tok/pos gömmesi, proj/fc2 ağırlık+sapma):
    varyans (d_eski/d_yeni)·s²·σ² = σ² olur, eps dahil aynen korunur. ln1/ln2'de eski γ /s (β aynı); ln_f'de
    γ /s², β /s (paylaşımlı çıkış tok·s ile çarpıldığından). Yeni boyutların normalize değeri tam 0, β_yeni = 0.
  * Yeni girdileri okuyan sütunlar (qkv/fc1'in yeni sütunları) ve yeni başlık/gizli birim satırları küçük rastgele
    (std 0.02) başlar: girdileri tam 0 olduğundan çıktıyı değiştirmez, simetriyi kırar. Yeni başlıkların ve yeni gizli
    birimlerin eski boyutlara yazdığı sütunlar (proj/fc2) SIFIR başlar: eğitim açana kadar etkisizdir.
  * Paylaşımlı çıkış (ln_f → tok.weight.T): ln_f'nin yeni boyutları 0 olduğundan tok'un yeni sütunları logite girmez.
Sonuç: fp32'de kaynak ile büyütülmüş modelin logitleri yuvarlama hatası içinde aynıdır (test: tests/test_grow.py).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .model import GptConfig, TinyGpt


def _check(src: GptConfig, dst: GptConfig) -> int:
    hd = src.d_model // src.n_head
    if dst.d_model // dst.n_head != hd or dst.d_model % dst.n_head:
        raise ValueError(f"başlık boyutu değişmemeli: {hd} → {dst.d_model / dst.n_head}")
    if dst.d_model < src.d_model or dst.d_ff < src.d_ff or dst.n_head < src.n_head:
        raise ValueError("yalnızca büyütme desteklenir")
    if (dst.vocab_size, dst.n_layer) != (src.vocab_size, src.n_layer):
        raise ValueError("sözlük ve katman sayısı aynı kalmalı (genişlik büyütme)")
    if dst.ctx < src.ctx:
        raise ValueError("bağlam küçülemez")
    return hd


@torch.no_grad()
def grow_state(sd: dict, src: GptConfig, dst: GptConfig, seed: int = 0, std: float = 0.02) -> dict:
    """Kaynak state_dict'ten büyütülmüş state_dict (fp32). Ayrıntı: modül açıklaması."""
    _check(src, dst)
    g = torch.Generator().manual_seed(seed)
    d0, d1, f0, f1 = src.d_model, dst.d_model, src.d_ff, dst.d_ff
    s = math.sqrt(d1 / d0)  # kalıntı akışı ölçeği (LN varyansı eps dahil korunur)

    def rnd(*shape):
        return torch.randn(*shape, generator=g) * std

    def emb(w: torch.Tensor, rows: int) -> torch.Tensor:
        # yeni sütunlar = eski sütunların satır ortalaması (kalıntı akışı ortalama izler)
        out = torch.zeros(rows, d1)
        w = w * s
        out[: w.shape[0], :d0] = w
        out[: w.shape[0], d0:] = w.mean(dim=1, keepdim=True)
        if rows > w.shape[0]:  # daha uzun bağlam: yeni konumlar küçük rastgele (eski konumlar aynen)
            out[w.shape[0]:] = rnd(rows - w.shape[0], d1)
        return out

    def ln(w: torch.Tensor, b: torch.Tensor, out_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        # out_scale: LN çıktısı bu oranla küçültülür (ln_f: 1/s, paylaşımlı çıkış tok·s'i dengeler)
        w1 = torch.empty(d1)
        w1[:d0] = w / s * out_scale
        w1[d0:] = w.abs().mean() / s * out_scale  # normalize değer 0 olduğundan başta etkisiz
        b1 = torch.zeros(d1)
        b1[:d0] = b * out_scale
        return w1, b1

    def residual_out(w: torch.Tensor, b: torch.Tensor, cols_new: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Kalıntıya yazan doğrusal katman (proj/fc2): eski satırlar eski sütunlarda aynen, yeni sütunlarda 0;
        # yeni satırlar eski satırların ortalaması (yeni boyut = eski boyutların ortalaması korunur)
        w1 = torch.zeros(d1, cols_new)
        w, b = w * s, b * s
        w1[:d0, : w.shape[1]] = w
        w1[d0:] = w1[:d0].mean(dim=0, keepdim=True)
        b1 = torch.zeros(d1)
        b1[:d0] = b
        b1[d0:] = b.mean()
        return w1, b1

    def reads_ln(w: torch.Tensor, b: torch.Tensor, rows_new: int, row_map) -> tuple[torch.Tensor, torch.Tensor]:
        # LN çıktısını okuyan katman (qkv/fc1): eski satırlar → row_map konumlarına; geri kalan her şey küçük rastgele
        w1 = rnd(rows_new, d1)
        b1 = torch.zeros(rows_new)
        for (s0, s1), (t0, t1) in row_map:
            w1[t0:t1, :d0] = w[s0:s1]
            b1[t0:t1] = b[s0:s1]
        return w1, b1

    out = {"tok.weight": emb(sd["tok.weight"].float(), dst.vocab_size),
           "pos.weight": emb(sd["pos.weight"].float(), dst.ctx)}
    for i in range(src.n_layer):
        p = f"blocks.{i}."
        f = {k[len(p):]: v.float() for k, v in sd.items() if k.startswith(p)}
        out[p + "ln1.weight"], out[p + "ln1.bias"] = ln(f["ln1.weight"], f["ln1.bias"])
        out[p + "ln2.weight"], out[p + "ln2.bias"] = ln(f["ln2.weight"], f["ln2.bias"])
        # qkv: [q | k | v], her biri başlık başına bitişik; eski başlıklar her parçanın başına
        qkv_map = [((j * d0, j * d0 + d0), (j * d1, j * d1 + d0)) for j in range(3)]
        out[p + "qkv.weight"], out[p + "qkv.bias"] = reads_ln(f["qkv.weight"], f["qkv.bias"], 3 * d1, qkv_map)
        out[p + "proj.weight"], out[p + "proj.bias"] = residual_out(f["proj.weight"], f["proj.bias"], d1)
        out[p + "fc1.weight"], out[p + "fc1.bias"] = reads_ln(f["fc1.weight"], f["fc1.bias"], f1, [((0, f0), (0, f0))])
        out[p + "fc2.weight"], out[p + "fc2.bias"] = residual_out(f["fc2.weight"], f["fc2.bias"], f1)
    out["ln_f.weight"], out["ln_f.bias"] = ln(sd["ln_f.weight"].float(), sd["ln_f.bias"].float(), out_scale=1 / s)
    return out


def grow_model(model: TinyGpt, dst: GptConfig, seed: int = 0) -> TinyGpt:
    """Büyütülmüş yeni TinyGpt (fp32, eval kipinde değil)."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    big = TinyGpt(dst)
    big.load_state_dict(grow_state(sd, model.config, dst, seed=seed))
    return big


def grow_from_dir(src_dir: Path, dst_cfg: dict, seed: int = 0) -> TinyGpt:
    """src_dir/model.bin (ORN1 int8) → float (export.load_quantized) → büyütülmüş model. dst_cfg: chat.SIZES[...]
    (vocab_size kaynaktan alınır)."""
    from .export import load_quantized
    src = load_quantized(Path(src_dir) / "model.bin")
    return grow_model(src, GptConfig(vocab_size=src.config.vocab_size, **dst_cfg), seed=seed)


def main(argv=None) -> None:
    from .chat import SIZES
    ap = argparse.ArgumentParser(description="ORN1 sohbet modelini genişlik olarak büyütür (ckpt.pt, adım 0)")
    ap.add_argument("--src", required=True, help="model.bin içeren klasör (ör. sohbet-kaggle-orta-0086000)")
    ap.add_argument("--size", choices=list(SIZES), default="buyuk")
    ap.add_argument("--out", required=True, help="yazılacak ckpt.pt (chat.train --resume-from ile kullanılabilir)")
    a = ap.parse_args(argv)
    m = grow_from_dir(Path(a.src), SIZES[a.size])
    torch.save({"config": m.config.to_dict(), "model": m.state_dict(), "step": 0}, a.out)
    print(f"büyütüldü: {m.n_params():,} parametre → {a.out}")


if __name__ == "__main__":
    main()
