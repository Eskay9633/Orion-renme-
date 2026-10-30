"""Tur 19: yeni beceri aileleri (yalnızca eğitim akışı) — geçerlilik, doğru sayı/zaman değerleri, çeşitlilik,
değerlendirme kümeleriyle örtüşmezlik ve ayrılmış (OOD) yapılardan uzaklık."""
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from orion_core import SHARED, datagen
from orion_core.protocol import validate_request

from test_datagen_tur18 import JACCARD_MAX, SPLIT_SHA, _closest, _Slots, _tokens  # noqa: F401

FAMILIES = [n for n, _ in datagen.SKILL_MIX]
FREE = ("text", "title", "query", "label")


def _samples(fam: str, n: int, seed: int) -> list[tuple[str, str, dict]]:
    rng = random.Random(seed)
    return [datagen.skill_example(rng, fam) for _ in range(n)]


@pytest.fixture(scope="module")
def samples() -> dict:
    return {f: _samples(f, 3000, 40 + i) for i, f in enumerate(FAMILIES)}


# --------------------------------------------------------------------------- geçerlilik ve değerler

def test_every_family_valid_and_args_copied_whole(samples):
    want_cmds = {"sayı": {"timer.start", "reminder.set"}, "göreli": {"reminder.set", "calendar.add"},
                 "uzun_ad": {"reminder.set", "calendar.add", "note.write"}, "not_arama": {"note.search"}}
    for fam, rows in samples.items():
        cmds = Counter()
        for cmd, src, args in rows:
            validate_request({"v": 1, "id": "g", "cmd": cmd, "args": args})
            cmds[cmd] += 1
            for k in FREE:
                if k in args:
                    # mastar belirtme hâliyle söylenebilir ("bakımını yaptırmayı" → "bakımını yaptırmak")
                    v = args[k]
                    assert v in src or (v.endswith(("mek", "mak")) and v[:-1] in src), (fam, src, args)
        assert set(cmds) == want_cmds[fam], (fam, cmds)


def test_number_words():
    assert datagen.num_words(15) == "on beş" and datagen.num_words(45) == "kırk beş"
    assert datagen.num_words(70) == "yetmiş" and datagen.num_words(9) == "dokuz" and datagen.num_words(25) == "yirmi beş"
    d = {nom: (m, at) for nom, m, at in datagen.SKILL_DURATIONS}
    assert d["on beş dakika"] == (15, "+15m") and d["kırk beş dakika"] == (45, "+45m")
    assert d["yarım saat"] == (30, "+30m") and d["çeyrek saat"] == (15, "+15m")
    assert d["bir buçuk saat"] == (90, "+90m") and d["1,5 saat"] == (90, "+90m") and d["iki buçuk saat"] == (150, "+150m")
    assert d["bir saat"] == (60, "+1h") and d["iki saat"] == (120, "+2h") and d["bir saat on beş dakika"] == (75, "+75m")
    assert datagen.dur_adj("yarım saat") == "yarım saatlik" and datagen.dur_adj("yirmi beş dakika") == "yirmi beş dakikalık"
    assert datagen.dur_dat("yarım saat") == "yarım saate" and datagen.dur_dat("on dakika") == "on dakikaya"
    assert datagen.with_ile("aidat") == "aidatla" and datagen.with_ile("vize başvurusu") == "vize başvurusuyla"
    assert datagen.with_ile("düğün") == "düğünle" and datagen.with_ile("bahçe") == "bahçeyle"


# Bağımsız başvuru çözümleyicisi (datagen'in kodunu kullanmaz): sayı sözcükleri + gün bölümü kuralları (Kotlin TimeParser).
_W = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
      "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50, "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90}


def _num(s: str) -> int:
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return int(s)
    return sum(_W[w] for w in s.split())


_NUM_RE = r"(\d+|(?:(?:on|yirmi|otuz|kırk|elli|altmış|yetmiş|seksen|doksan)(?: (?:bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz))?)|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz)"


def _ref_minutes(nom: str) -> int:
    if nom == "yarım saat":
        return 30
    if nom == "çeyrek saat":
        return 15
    if m := re.fullmatch(r"(1,5) saat", nom):
        return 90
    if m := re.fullmatch(_NUM_RE + r" buçuk saat", nom):
        return _num(m.group(1)) * 60 + 30
    if m := re.fullmatch(_NUM_RE + r" saat " + _NUM_RE + r" dakika", nom):
        return _num(m.group(1)) * 60 + _num(m.group(2))
    if m := re.fullmatch(_NUM_RE + r" saat", nom):
        return _num(m.group(1)) * 60
    m = re.fullmatch(_NUM_RE + r" dakika", nom)
    return _num(m.group(1))


def test_durations_match_reference():
    for nom, minutes, at in datagen.SKILL_DURATIONS:
        assert _ref_minutes(nom) == minutes, nom
        assert at in (f"+{minutes}m", f"+{minutes // 60}h") and (at.endswith("m") or minutes % 60 == 0), (nom, at)


def _ref_clock(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(sabah|öğlen|öğleden sonra|akşam|gece)(?: (.+?)(?:'(?:de|da|te|ta)|(?<=[a-zçğıöşü])(?:de|da|te|ta))?( buçukta)?)?", s)
    assert m, s
    part, n, half = m.group(1), m.group(2), m.group(3)
    if part == "öğlen":
        return 12, 30 if half else 0
    if n.endswith("buçukta"):
        n, half = n[: -len(" buçukta")], True
    words = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
             "on bir": 11, "on iki": 12}
    h = int(n) if n.isdigit() else words[n]
    if part in ("öğleden sonra", "akşam") and h <= 11 or part == "gece" and 6 <= h <= 11:
        h += 12
    return h, 30 if half else 0


def _ref_when(tr: str) -> str:
    if m := re.fullmatch(_NUM_RE + r" saat sonra", tr):
        return f"+{_num(m.group(1))}h"
    if m := re.fullmatch(_NUM_RE + r" gün sonra", tr):
        return f"+{_num(m.group(1))}d"
    if m := re.fullmatch(_NUM_RE + r" gün sonra (.+)", tr):
        h, mm = _ref_clock(m.group(2))
        return f"d+{_num(m.group(1))}@{h:02d}:{mm:02d}"
    day = {"bugün": "d+0", "yarın": "d+1", "öbür gün": "d+2", "bu": "d+0", **{d: c for d, c in datagen.DOW_TR}}
    for k in sorted(day, key=len, reverse=True):
        if tr.startswith(k + " "):
            h, mm = _ref_clock(tr[len(k) + 1:])
            return f"{day[k]}@{h:02d}:{mm:02d}"
    h, mm = _ref_clock(tr)
    return f"d+0@{h:02d}:{mm:02d}"


def test_relative_times_match_reference():
    rng = random.Random(8)
    seen = Counter()
    for _ in range(4000):
        t = datagen._rel_time(rng, rng.random() < 0.7)
        assert _ref_when(t.tr) == t.when, (t.tr, t.when)
        seen[t.when[0] if t.when[0] == "+" else t.when.split("@")[0][:2]] += 1
        validate_request({"v": 1, "id": "g", "cmd": "reminder.set", "args": {"at": t.when, "text": "x"}})
    # Ofset, gün+saat, haftanın günü, bugün hepsi var
    assert seen["+"] > 300 and seen["d+"] > 1000 and sum(v for k, v in seen.items() if k not in ("+", "d+")) > 100, seen
    trs = {datagen._rel_time(rng, True).tr for _ in range(3000)}
    for want in ("üç saat sonra", "iki gün sonra", "bu akşam sekizde", "öğleden sonra 4'te"):
        assert any(want in s for s in trs), want


def test_calendar_never_gets_offsets(samples):
    for fam in ("göreli", "uzun_ad"):
        for cmd, _src, args in samples[fam]:
            if cmd == "calendar.add":
                assert not args["start"].startswith("+"), args


# --------------------------------------------------------------------------- çeşitlilik ve paylar

def test_variety(samples):
    for fam, rows in samples.items():
        srcs = [s for _, s, _ in rows]
        assert len(set(srcs)) / len(srcs) > {"sayı": 0.65, "not_arama": 0.55}.get(fam, 0.85), fam
    num = [s for c, s, _ in samples["sayı"] if c == "timer.start"]
    assert sum(any(f in s for f in ("dakikalık", "saatlik")) for s in num) > 300
    assert sum(any(f in s for f in ("dakikaya", "saate")) for s in num) > 200
    assert any("buçuk" in s for s in num) and any("yarım saat" in s for s in num)
    ns = [s for _, s, _ in samples["not_arama"]]
    for f in ("ile ilgili notlarımı bulur musun", "hakkındaki notlar", "yla ilgili", "la ilgili", "ile alakalı"):
        assert any(f in s for s in ns), f
    mw = {a.get("text") or a.get("title") for _, _, a in samples["uzun_ad"]}
    assert {"annemin doğum günü", "kombi servisi randevusu"} <= mw
    assert all(len(x.split()) >= 2 for x in mw)


def test_shares_documented():
    """t2c eğitim çekilişlerinin %11'i; tüm akışta (tohum %3, dış veri %5, t2c %80) ≈ %8,1 — istenen aralık %8–12."""
    from orion_core.massive import EXT_FRAC
    assert EXT_FRAC == 0.05
    assert datagen.SKILL_FRAC == 0.11 and abs(sum(w for _, w in datagen.SKILL_MIX) - 1) < 1e-9
    whole = (1 - 0.03) * (1 - EXT_FRAC) * 0.8 * datagen.SKILL_FRAC
    assert 0.08 <= whole <= 0.12, whole
    # Diğer paylar düşürülmedi
    assert datagen.CONTRAST_FRAC == 0.08 and datagen.IDENTITY_FRAC == 0.01


def test_families_reach_training_stream_only():
    counted = Counter()
    real = datagen.skill_example

    def spy(rng, family=None):
        counted["n"] += 1
        return real(rng, family)

    datagen.skill_example = spy
    try:
        datagen.t2c_set(10000, "train", seed=19)
        assert abs(counted["n"] / 10000 - datagen.SKILL_FRAC) < 0.015, counted
        counted.clear()
        for split in ("test_iid", "test_ood", "legacy_ood"):
            datagen.t2c_set(500, split, seed=5)
        assert counted["n"] == 0
    finally:
        datagen.skill_example = real


# --------------------------------------------------------------------------- örtüşmezlik

def _norm(s: str) -> str:
    return datagen._fence_norm(s)


def _toks(s: str) -> frozenset:
    return datagen._fence_toks(s)


def _eval_sentences() -> dict[str, list[str]]:
    out: dict = {}
    for name in ("handwritten-eval.jsonl", "blind-eval.jsonl"):
        out[name] = [json.loads(x)["src"] for x in (SHARED / "core-model" / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    out["cognitive-exam"] = [t for x in (SHARED / "core-model" / "cognitive-exam.jsonl").read_text(encoding="utf-8").splitlines()
                             if x.strip() for t in json.loads(x)["turns"]]
    for split, n, seed in SPLIT_SHA:  # test_iid/test_ood/legacy_ood + kalibrasyon kümeleri
        out[f"{split}-{seed}"] = [e["src"] for e in datagen.t2c_set(n, split, seed)]
    return out


def _leaks(srcs: list[str], refs: dict[str, list[str]]) -> list:
    bad = []
    for name, sents in refs.items():
        norm = {_norm(s) for s in sents}
        by_len: dict = {}
        for s in sents:
            t = _toks(s)
            by_len.setdefault(len(t), {}).setdefault(t, s)
        for src in set(srcs):
            if _norm(src) in norm:
                bad.append((name, src, "aynı"))
                continue
            t = _toks(src)
            for n in range(int(len(t) * 0.8), int(len(t) / 0.8) + 1):
                for u, s in by_len.get(n, {}).items():
                    if len(t & u) / len(t | u) >= 0.8:
                        bad.append((name, src, s))
    return bad


def test_no_eval_sentence_or_near_duplicate_in_families():
    refs = _eval_sentences()
    srcs = [s for i, f in enumerate(FAMILIES) for _, s, _ in _samples(f, 2500, 70 + i)]
    bad = _leaks(srcs, refs)
    assert not bad, bad[:10]
    # ölçü yakalıyor: el yazımı cümlenin kendisi ve tek sözcük eklenmiş hâli
    hw = refs["handwritten-eval.jsonl"][-1]
    assert _leaks([hw, hw + " hemen"], {"hw": refs["handwritten-eval.jsonl"]})


def test_fence_rejects_eval_sentences():
    assert datagen._fenced("on beş dakika geri sayım başlat") and datagen._fenced("lütfen on beş dakika geri sayım başlat")
    assert datagen._fenced("notlarda veteriner geçenleri göster")
    assert not datagen._fenced("kırk beş dakikalık pilav sayacı kur")


_EXT = Path(os.environ["ORION_EXT_DATA"]) if os.environ.get("ORION_EXT_DATA") else None


@pytest.mark.skipif(not (_EXT and (_EXT / "massive_tr.jsonl").exists()), reason="gerçek MASSIVE/xSID hazırlanmadı (ağsız test)")
def test_no_massive_xsid_eval_sentence_in_families():
    refs = {n: [json.loads(x)["src"] for x in (_EXT / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            for n in ("massive_tr", "xsid_tr")}
    srcs = [s for i, f in enumerate(FAMILIES) for _, s, _ in _samples(f, 2500, 90 + i)]
    bad = _leaks(srcs, refs)
    assert not bad, bad[:10]


def _eval_values() -> set:
    vals = set()
    for name in ("handwritten-eval.jsonl", "blind-eval.jsonl"):
        for x in (SHARED / "core-model" / name).read_text(encoding="utf-8").splitlines():
            if x.strip():
                a = json.loads(json.loads(x)["tgt"])["args"]
                vals |= {v for k, v in a.items() if k in FREE + ("topic", "question")}
    return vals


def test_value_pools_disjoint_from_eval_and_held_slices():
    held = set()
    for pool in (datagen.TIMER_LABELS, datagen.REMINDER_TEXTS, datagen.EVENT_TITLES, datagen.NOTE_TEXTS, datagen.QUERIES):
        held |= set(pool[max(1, int(len(pool) * 0.75)):])
    ev = _eval_values()
    for pool in (datagen.SKILL_TIMER_LABELS, datagen.MULTI_REM_TEXTS, datagen.MULTI_EVENT_TITLES, datagen.MULTI_NOTE_TEXTS,
                 datagen.SKILL_QUERIES):
        assert not set(pool) & held, set(pool) & held
        assert not set(pool) & ev, set(pool) & ev
    # El yazımı sorgular (kira, veteriner, kapıcı, doğum günü) ve kör sorgular (sigorta, anahtar) sorgu havuzunda yok
    assert not {"kira", "veteriner", "kapıcı", "doğum günü", "sigorta", "anahtar"} & set(datagen.SKILL_QUERIES)


# --------------------------------------------------------------------------- ayrılmış yapılar

@pytest.fixture(scope="module")
def held_out() -> dict:
    """test_datagen_tur18.held_out ile aynı kaynak: test_ood dilbilgisi örnekleri + legacy_ood kalıpları, yuvalarla."""
    rng = random.Random(1)
    out: dict = {}
    with _Slots():
        for cmd in datagen.GRAMMAR:
            for _ in range(1500):
                try:
                    tr, _a = datagen.grammar_example(rng, cmd, "test_ood")
                except Exception:
                    continue
                out.setdefault(_tokens(tr), tr)
            for _ in range(200):
                try:
                    tr, _a = rng.choice(datagen._templates_for(cmd, "test_ood"))(rng, "test_ood")
                except Exception:
                    continue
                out.setdefault(_tokens(tr), tr)
    assert len(out) > 3000
    return out


def test_families_far_from_held_out_structures(held_out):
    """test_datagen_tur18 ile aynı ölçü: yuvalar normalize edildikten sonra hiçbir aile cümlesi ayrılmış (test_ood /
    legacy_ood) bir yapıya sözcük-Jaccard ≥ 0.6 ile benzemez."""
    rng = random.Random(3)
    bad = []
    with _Slots():
        for fam in FAMILIES:
            for _ in range(400):
                _cmd, tr, _a = datagen.skill_example(rng, fam)
                score, h = _closest(tr, held_out)
                if score >= JACCARD_MAX:
                    bad.append((fam, tr, h, round(score, 2)))
    assert not bad, sorted(set(bad))[:15]


def test_families_use_no_held_out_parts():
    """Ayrılmış fiil/çerçeve parçaları (split_pool 'held') hiçbir aile cümlesinde geçmez."""
    G = datagen.G
    held = set()
    for pool in (datagen.REM_VERBS, datagen.CAL_VERBS, datagen.TIMER_VERBS, datagen.NOTE_VERBS, datagen.NS_VERBS):
        held |= set(G.split_pool(pool, "held"))
    rng = random.Random(4)
    for fam in FAMILIES:
        for _ in range(1500):
            _c, src, _a = datagen.skill_example(rng, fam)
            for h in held:
                assert not re.search(r"(^|[\s,:])" + re.escape(h) + r"($|[\s,:])", src), (h, src)
