"""Elle yazılmış değerlendirme kümesi: biçim, protokol geçerliliği, kapsam ve eğitim verisiyle örtüşmezlik."""
import hashlib
import json
import random
import re
from collections import Counter

from orion_core import SHARED, datagen
from orion_core.evaluate import load_handwritten
from orion_core.protocol import canonical, load_schema, validate_request

PATH = SHARED / "core-model" / "handwritten-eval.jsonl"
# İlk 29 cümle (Tur 13'ten beri) değişmeden kalır; yeni cümleler yalnızca SONA eklenir.
ORIGINAL_N = 29
ORIGINAL_SHA = "3d43064e16f85cd7ba5718f869a3052acbe853b5dceb1a103e1d7c7c3eecba9d"
REQUIRED = ("task", "src", "tgt", "provider", "model", "date", "license", "training_allowed", "origin")


def _norm(s: str) -> str:
    """Nezaket ön/son eklerini ve noktalamayı atıp karşılaştırır (eğitimde aynı cümlenin süslü hâli de sayılır)."""
    s = s.lower().strip().rstrip(".?!").strip()
    for p in ("hey orion ", "orion, ", "orion ", "lütfen ", "acaba "):
        if s.startswith(p):
            s = s[len(p):]
    for p in (" lütfen", " teşekkürler", " rica etsem"):
        if s.endswith(p):
            s = s[: -len(p)]
    return s.strip()


def test_original_items_unchanged():
    lines = PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    assert hashlib.sha256("".join(lines[:ORIGINAL_N]).encode("utf-8")).hexdigest() == ORIGINAL_SHA


def test_every_item_is_valid_canonical_command():
    rows = load_handwritten()
    assert len(rows) >= 100
    for r in rows:
        for k in REQUIRED:
            assert k in r, (k, r["src"])
        assert r["task"] == "t2c"
        t = json.loads(r["tgt"])
        assert set(t) == {"args", "cmd"}, r["src"]
        validate_request({"v": 1, "id": "g", "cmd": t["cmd"], "args": t["args"]})
        # Hedef, yalnızca söylenen argümanlarla kanonik biçimde (değerlendirme bunu tam istekle karşılaştırır)
        assert r["tgt"] == canonical({"cmd": t["cmd"], "args": t["args"]}), r["src"]
    srcs = [r["src"] for r in rows]
    assert len(set(srcs)) == len(srcs)
    assert len({_norm(s) for s in srcs}) == len(srcs)


def test_new_items_are_eval_only():
    for r in load_handwritten()[ORIGINAL_N:]:
        assert r["training_allowed"] is False and r.get("eval_only") is True and r["origin"] == "local", r["src"]


def test_covers_every_command():
    cmds = Counter(json.loads(r["tgt"])["cmd"] for r in load_handwritten())
    assert set(cmds) == set(load_schema().commands), set(load_schema().commands) - set(cmds)
    # En sık kullanılan komutlar daha çok örnekle temsil edilir.
    assert cmds["reminder.set"] >= max(v for k, v in cmds.items() if k != "reminder.set")


def _reference_sentences() -> list[str]:
    """Tohum eğitim kümesi, kör küme, büyük bir eğitim akışı örneği ve karşıt çiftler."""
    out = []
    for name in ("seed-train.jsonl", "blind-eval.jsonl"):
        for x in (SHARED / "core-model" / name).read_text(encoding="utf-8").splitlines():
            if x.strip():
                out.append(json.loads(x)["src"])
    for seed in (11, 12, 13):
        out += [e["src"] for e in datagen.t2c_set(30000, "train", seed=seed)]
    rng = random.Random(5)
    out += [datagen.contrast_example(rng)[1] for _ in range(20000)]
    return out


def test_new_items_not_in_training_data():
    """Yeni cümleler tohum eğitim kümesinde, kör kümede, karşıt çiftlerde ve büyük bir eğitim akışı örneğinde yok."""
    new = {_norm(r["src"]): r["src"] for r in load_handwritten()[ORIGINAL_N:]}
    seen = {_norm(s) for s in _reference_sentences()}
    hit = [src for k, src in new.items() if k in seen]
    assert not hit, hit


def _toks(s: str) -> frozenset:
    return frozenset(re.sub(r"[.,:;?!\"]", " ", _norm(s)).split())


def test_new_items_not_near_duplicates_of_training_data():
    """Birebir değil ama neredeyse aynı da değil: yeni cümlelerin hiçbiri eğitim/kör/karşıt cümlelerine sözcük-Jaccard
    >= 0.8 ile benzemez ("bugün bir işim var mı" ~ eğitimdeki "bugün işim var mı" ölçümü şişirirdi)."""
    by_len: dict = {}
    for s in _reference_sentences():
        t = _toks(s)
        by_len.setdefault(len(t), {}).setdefault(t, s)
    bad = []
    for r in load_handwritten()[ORIGINAL_N:]:
        t = _toks(r["src"])
        for n in range(int(len(t) * 0.8), int(len(t) / 0.8) + 1):  # |A∩B|/|A∪B| >= 0.8 ⇒ boyut oranı >= 0.8
            for u, s in by_len.get(n, {}).items():
                if len(t & u) / len(t | u) >= 0.8:
                    bad.append((r["src"], s))
    assert not bad, bad[:10]
    # ölçü gerçekten yakalıyor
    a, b = _toks("bugün bir işim var mı"), _toks("bugün işim var mı")
    assert len(a & b) / len(a | b) >= 0.8
