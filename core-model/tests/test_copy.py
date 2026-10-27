import json

from orion_core import datagen
from orion_core.bpe import BPE
from orion_core.copy import N_PTR, decode_output, encode_target, mark_target
from orion_core.lexicon import lexicon_sentences


def _bpe():
    ex = datagen.generate(3000, "train", seed=1)
    return BPE.train([e["src"] for e in ex] + [e["tgt"] for e in ex] + lexicon_sentences(1000), 1500)


def test_pointer_roundtrip_and_coverage():
    bpe = _bpe()
    assert bpe.to_json()["pointers"] == N_PTR
    n = used = 0
    for split in ("train", "test_iid", "test_ood"):
        for e in datagen.t2c_set(400, split, seed=7):
            src = bpe.encode(e["src"])
            ids = encode_target(bpe, src, e["tgt"], "t2c")
            assert decode_output(bpe, src, ids) == e["tgt"], e
            n += 1
            used += any(i >= bpe.vocab_size for i in ids)
    assert used / n > 0.6  # serbest metinli komutların çoğu işaretçiyle kopyalanmalı


def test_marks_only_verbatim_values():
    bpe = _bpe()
    src = bpe.encode("not al: Ayşe'nin doğum günü 3 Mart")
    t = mark_target(bpe, src, json.dumps({"cmd": "note.write", "args": {"text": "Ayşe'nin doğum günü 3 Mart"}}))
    assert "" in t
    t2 = mark_target(bpe, src, json.dumps({"cmd": "note.write", "args": {"text": "başka bir şey"}}))
    assert "" not in t2
