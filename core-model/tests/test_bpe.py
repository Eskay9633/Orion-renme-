from orion_core import datagen
from orion_core.bpe import BPE, FIRST_MERGE, pretokenize


def _corpus(n=3000):
    ex = datagen.generate(n, "train", seed=11)
    return [e["src"] for e in ex] + [e["tgt"] for e in ex]


def test_roundtrip_and_compression():
    texts = _corpus()
    bpe = BPE.train(texts, vocab_size=1200)
    assert bpe.vocab_size <= 1200
    for t in texts[:300] + ["çğıöşü ÇĞİÖŞÜ 🙂 {\"a\":1}", "", "  iki  boşluk  "]:
        assert bpe.decode(bpe.encode(t)) == t
    raw = sum(len(t.encode()) for t in texts[:300])
    tok = sum(len(bpe.encode(t)) for t in texts[:300])
    assert tok < raw / 2.5, (tok, raw)


def test_digits_are_single_tokens():
    assert pretokenize("saat 10:30'da") == ["saat", " 1", "0", ":", "3", "0", "'", "da"]


def test_deterministic_training():
    texts = _corpus(800)
    assert BPE.train(texts, 600).merges == BPE.train(texts, 600).merges


def test_save_load(tmp_path):
    bpe = BPE.train(_corpus(500), 500)
    p = tmp_path / "tok.json"
    bpe.save(p)
    b2 = BPE.load(p)
    assert b2.merges == bpe.merges
    assert b2.encode("yarın saat 9'da toplantı") == bpe.encode("yarın saat 9'da toplantı")
    assert min(i for i in b2.encode("toplantı toplantı")) >= 8
    assert any(i >= FIRST_MERGE for i in b2.encode("toplantı"))


def test_kv_cache_greedy_matches_plain_greedy():
    import torch
    from orion_core.bpe import EOS, PAD
    from orion_core.model import GptConfig, TinyGpt
    torch.manual_seed(0)
    m = TinyGpt(GptConfig(vocab_size=300, ctx=40, d_model=32, n_layer=2, n_head=2, d_ff=64))
    prompts = [[1, 4, 50, 60, 3], [1, 4, 70, 3], [1, 4, 80, 90, 3], [1, 4, 7, 8, 9, 10, 3]]
    fast = m.greedy_batch(prompts, EOS, PAD, max_new=20)
    slow = [m.greedy(p, EOS, max_new=20) for p in prompts]
    assert fast == slow
