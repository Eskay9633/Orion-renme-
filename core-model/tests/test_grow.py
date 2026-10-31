"""Genişlik büyütme (grow.py): küçük yapay yapılandırmayla kaynak ve büyütülmüş modelin aynı işlevi verdiği.

Tolerans: fp32'de en büyük mutlak logit farkı < 1e-4 (yöntem eps dahil tam; fark yalnızca toplama sırası).
Gerçek 86000 modelinde (yerel ölçüm, eğitimsiz): logit farkı 5.8e-5, ppl 19.0105 → 19.0105.
"""
import math
import random

import pytest
import torch
import torch.nn.functional as F

from orion_core.grow import grow_model, grow_state
from orion_core.model import GptConfig, TinyGpt

SRC = GptConfig(vocab_size=97, ctx=24, d_model=32, n_layer=2, n_head=4, d_ff=64)   # başlık boyutu 8
DST = GptConfig(vocab_size=97, ctx=24, d_model=48, n_layer=2, n_head=6, d_ff=96)   # orta → büyük oranı (1.5×)


def _trained_like(seed=0) -> TinyGpt:
    """Rastgele ama "eğitilmiş gibi" (LN γ/β ve sapmalar sıfırdan farklı, bazı boyutlar baskın) kaynak model."""
    torch.manual_seed(seed)
    m = TinyGpt(SRC)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(".bias"):
                p.normal_(0, 0.1)
            elif "ln" in n:
                p.uniform_(0.5, 1.5)
            else:
                p.mul_(5)
        m.tok.weight[:, 3] += 0.5  # kalıntı ortalaması sıfırdan uzak (LN ortalama durumu sınanır)
    return m.eval()


def _prompts():
    rng = random.Random(1)
    return [torch.tensor([[rng.randrange(SRC.vocab_size) for _ in range(n)]]) for n in (1, 7, 24)]


def test_grown_logits_match_source():
    src = _trained_like()
    big = grow_model(src, DST).eval()
    assert big.n_params() > src.n_params()
    with torch.no_grad():
        for x in _prompts():
            assert (src(x) - big(x)).abs().max().item() < 1e-4


def test_grown_ppl_and_greedy_match():
    src = _trained_like(seed=3)
    big = grow_model(src, DST).eval()
    x = _prompts()[2]
    with torch.no_grad():
        a = F.cross_entropy(src(x[:, :-1])[0], x[0, 1:]).item()
        b = F.cross_entropy(big(x[:, :-1])[0], x[0, 1:]).item()
    assert math.exp(a) == pytest.approx(math.exp(b), rel=1e-4)
    assert src.greedy([1, 2, 3], eos=0, max_new=10) == big.greedy([1, 2, 3], eos=0, max_new=10)


def test_new_dims_trainable_and_seeded():
    src = _trained_like()
    a = grow_state(src.state_dict(), SRC, DST, seed=5)
    b = grow_state(src.state_dict(), SRC, DST, seed=5)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # yeni başlıkların q/k/v satırları rastgele (simetri kırılır), eski boyutlara yazan sütunları sıfır
    assert a["blocks.0.qkv.weight"][SRC.d_model:DST.d_model, :SRC.d_model].abs().sum() > 0
    assert a["blocks.0.proj.weight"][:SRC.d_model, SRC.d_model:].abs().sum() == 0
    assert a["blocks.0.fc2.weight"][:SRC.d_model, SRC.d_ff:].abs().sum() == 0
    # büyütülmüş model bir adım eğitilebilir; sonra yeni sütunlar sıfırdan ayrılır
    big = grow_model(src, DST)
    opt = torch.optim.SGD(big.parameters(), lr=0.1)
    x = _prompts()[2]
    F.cross_entropy(big(x[:, :-1])[0], x[0, 1:]).backward()
    opt.step()
    assert big.blocks[0].proj.weight[:SRC.d_model, SRC.d_model:].abs().sum() > 0


def test_rejects_head_dim_change():
    with pytest.raises(ValueError):
        grow_state(TinyGpt(SRC).state_dict(), SRC, GptConfig(vocab_size=97, ctx=24, d_model=48, n_layer=2, n_head=4, d_ff=96))


def test_train_grown_with_teacher(tmp_path):
    """chat.train: --init-grown-from (ckpt yokken büyüt, adım 0) + --teacher (yalnız diyalog satırlarında damıtma);
    devamda büyütme yok sayılır; tokenizer farklıysa hata."""
    import shutil

    from orion_core import chat
    from orion_core.export import export
    texts = [f"Orion telefonda yaşayan bir yapay zekadır. Bugün {i}. gün ve hava güzel." * 3 for i in range(200)]
    data = tmp_path / "data"
    chat.prepare(data, 10**9, vocab=300, texts=texts, open_data=False)
    small = dict(ctx=48, d_model=32, n_layer=1, n_head=2, d_ff=64)
    chat.train(data, tmp_path / "t-run", steps=10, budget_min=5, batch=4, lr=3e-3, dialog_frac=0.5, config=small)
    tdir = tmp_path / "teacher"
    tdir.mkdir()
    export(tmp_path / "t-run" / "ckpt.pt", tdir / "model.bin")
    shutil.copy(data / "tokenizer.json", tdir / "tokenizer.json")
    big = dict(ctx=48, d_model=48, n_layer=1, n_head=3, d_ff=96)
    run = tmp_path / "run"
    s = chat.train(data, run, steps=6, budget_min=5, batch=4, lr=1e-3, dialog_frac=0.5, config=big,
                   teacher=tdir, init_grown_from=tdir)
    assert s["step"] == 6 and s["init_grown_from"] == str(tdir)
    assert s["distill"]["T"] == 2.0 and s["distill"]["alpha"] == 0.5 and s["distill"]["kd_loss"] >= 0
    assert s["params"] > 0 and s["val_loss"] == s["val_loss"]
    s2 = chat.train(data, run, steps=8, budget_min=5, batch=4, lr=1e-3, dialog_frac=0.5, config=big,
                    teacher=tdir, init_grown_from=tdir)
    assert s2["step"] == 8  # devam: büyütme yeniden yapılmaz
    (tdir / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        chat.train(data, tmp_path / "run2", steps=2, budget_min=5, batch=2, lr=1e-3, dialog_frac=0.5, config=big,
                   teacher=tdir)


def test_kd_loss_zero_when_equal_and_positive_otherwise():
    from orion_core.chat import kd_loss
    a = torch.randn(2, 5, 11)
    assert kd_loss(a, a).abs().max() < 1e-5
    b = torch.randn(2, 5, 11)
    assert (kd_loss(a, b) > 0).all() and (kd_loss(a, b, reverse=True) > 0).all()
