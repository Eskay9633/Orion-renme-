"""Girdiden kopyalama (işaretçi) mekanizması.

Küçük model serbest metni (not, başlık, konu...) harf harf üretirken bozuyordu. Bunun yerine
hedefte serbest metin değeri, girdinin BELİRTEÇ konumlarına işaret eden iki özel belirteçle
yazılır: <P i><P j> = girdinin i. belirtecinden j. belirtecine kadar olan metin (baştaki tek
boşluk atılır). Dikkat mekanizması konumu doğrudan gördüğü için bunu öğrenmek kolaydır.

Kimlikler: BPE sözlüğünün hemen arkasından N_PTR tane (bpe.vocab_size + i).
Kotlin karşılığı: orion-core/.../model/CoreModel.kt (aynı kural).
"""
from __future__ import annotations

import json

import regex as re

from .bpe import BPE

N_PTR = 96
COPY_KEYS = ("text", "title", "topic", "query", "label", "place", "need", "prompt", "question", "repo", "coin", "name", "input", "goal", "role", "to", "subject", "body")
_MARK = re.compile("(\\d+)(\\d+)")


def span_text(bpe: BPE, src_ids: list[int], i: int, j: int) -> str:
    t = bpe.decode(src_ids[i:j + 1])
    return t[1:] if t.startswith(" ") else t


def find_span(bpe: BPE, src_ids: list[int], value: str) -> tuple[int, int] | None:
    if not value:
        return None
    n = min(len(src_ids), N_PTR)
    for i in range(n):
        acc = ""
        for j in range(i, n):
            acc = span_text(bpe, src_ids, i, j)
            if acc == value:
                return i, j
            if len(acc) > len(value) + 1 or not value.startswith(acc[: len(value)].rstrip("�")):
                break
    return None


def copy_marker(bpe: BPE, src_ids: list[int], value: str) -> str | None:
    """Değerin girdideki karşılığı: tam aralık ya da (mastar için) kök aralığı + 'k'.
    'faturayı ödemeyi hatırlat' → text 'faturayı ödemek' = <P i><P j> ('faturayı ödeme') + 'k'."""
    sp = find_span(bpe, src_ids, value)
    if sp:
        return f"\ue000{sp[0]}\ue001{sp[1]}\ue002"
    if value.endswith(("mek", "mak")) and len(value) > 4:
        sp = find_span(bpe, src_ids, value[:-1])
        if sp:
            return f"\ue000{sp[0]}\ue001{sp[1]}\ue002k"
    return None


def mark_answer(bpe: BPE, src_ids: list[int], tgt: str) -> str:
    """Cevap görevinde tırnak içindeki değerler (ör. hatırlatıcı metni) girdi JSON'undan kopyalanır."""
    def rep(m):
        mk = copy_marker(bpe, src_ids, m.group(1))
        return f'"{mk}"' if mk else m.group(0)
    return re.sub('"([^"\ue000]{1,200})"', rep, tgt)


def mark_target(bpe: BPE, src_ids: list[int], tgt: str) -> str:
    """t2c hedefindeki kopyalanabilir değerleri işaretleyicilerle değiştirir (kanonik JSON korunur)."""
    try:
        obj = json.loads(tgt)
    except json.JSONDecodeError:
        return tgt
    if not isinstance(obj, dict) or not isinstance(obj.get("args"), dict):
        return tgt
    args = dict(obj["args"])
    for k in COPY_KEYS:
        v = args.get(k)
        if isinstance(v, str):
            mk = copy_marker(bpe, src_ids, v)
            if mk:
                args[k] = mk
    obj["args"] = args
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def encode_target(bpe: BPE, src_ids: list[int], tgt: str, task: str) -> list[int]:
    text = mark_target(bpe, src_ids, tgt) if task == "t2c" else mark_answer(bpe, src_ids, tgt) if task == "ans" else tgt
    out: list[int] = []
    pos = 0
    base = bpe.vocab_size
    for m in _MARK.finditer(text):
        out += bpe.encode(text[pos:m.start()])
        out += [base + int(m.group(1)), base + int(m.group(2))]
        pos = m.end()
    out += bpe.encode(text[pos:])
    return out


def _infinitive_k(span: str, rest: str) -> str:
    """Mastar işaretçisi '+k' yalnızca -me/-ma ile biten aralığa eklenir ('ödeme'+k → 'ödemek').
    Başka aralıktan sonra gelen 'k' (ör. 'markete'+k, 'ilaç'+k) atılır. Kotlin: CoreModel.decodeOutput (aynı kural)."""
    if rest.startswith("k") and (len(rest) == 1 or rest[1] == '"') and not span.lower().endswith(("me", "ma")):
        return rest[1:]
    return rest


def decode_output(bpe: BPE, src_ids: list[int], ids: list[int]) -> str:
    """Model çıktısını çözer: işaretçi çiftleri girdideki metinle değiştirilir."""
    base = bpe.vocab_size
    parts: list[str] = []
    buf: list[int] = []
    span: str | None = None  # son eklenen aralık (ardından gelen metin '+k' için denetlenir)

    def flush():
        nonlocal span
        t = bpe.decode(buf)
        if span is not None:
            t = _infinitive_k(span, t)
        parts.append(t)
        buf.clear()
        span = None

    i = 0
    while i < len(ids):
        t = ids[i]
        if t >= base:
            flush()
            a = t - base
            if i + 1 < len(ids) and ids[i + 1] >= base:
                b = ids[i + 1] - base
                i += 2
            else:
                b = a
                i += 1
            if a <= b < len(src_ids):
                span = span_text(bpe, src_ids, a, b)
                parts.append(span)
            continue
        buf.append(t)
        i += 1
    flush()
    return "".join(parts)
