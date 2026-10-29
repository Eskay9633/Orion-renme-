"""Tur 14 veri düzeltmeleri: uzun cevap metinleri, karşıt çiftler, yeni saat söyleyişleri, hesap, beceri hedefi.
Ayrıca sızıntı yok: karşıt çiftler ayrılmış parçaları ve legacy_ood/el yazımı/kör kümeleri kullanmaz."""
import json
import random
import re
from collections import Counter

from orion_core import SHARED, datagen
from orion_core import grammar as G


def _ans_reminders(split: str, n: int = 6000, seed: int = 3) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        e = datagen.answer_example(rng, split)
        res = json.loads(e["src"])["result"]
        if isinstance(res, dict) and res.get("kind") == "reminder":
            assert e["tgt"] == f"Tamam, \"{res['text']}\" için hatırlatıcı kurdum."
            out.append(res["text"])
    return out


# --------------------------------------------------------------------------- 1. cevap metinleri

def test_answer_reminder_texts_are_long_and_varied_in_train():
    xs = _ans_reminders("train")
    multi = sum(len(x.split()) >= 2 for x in xs) / len(xs)
    assert multi > 0.6, multi
    assert sum(len(x.split()) >= 4 for x in xs) / len(xs) > 0.15
    assert len(set(xs)) / len(xs) > 0.8  # 30 sabit metin değil


def test_answer_reminder_texts_train_test_disjoint():
    held = set(datagen._split(datagen.REMINDER_TEXTS, "test_iid"))
    assert set(_ans_reminders("train")).isdisjoint(held)
    assert set(_ans_reminders("test_iid", 2000)) <= held  # değerlendirme değişmedi: ayrılmış listeden


# --------------------------------------------------------------------------- 2. karşıt çiftler

def _contrast(n: int = 6000, seed: int = 9) -> list[tuple[str, str, dict]]:
    rng = random.Random(seed)
    return [datagen.contrast_example(rng) for _ in range(n)]


EXPECT = [  # (cümle örüntüsü, komut)
    (r"^tercihimi kaydet: ", "memory.save"), (r"^hafızana kaydet: ", "memory.save"),
    (r"^bunu tercih olarak kaydet: ", "memory.save"), (r"kayda geç", "note.write"), (r"^kaydet: ", "note.write"),
    (r"^not olarak kaydet: ", "note.write"),
    (r"^\d+ dakika sonra haber ver diye ", "timer.start"), (r" için \d+ dakika (zamanlayıcı kur|sayaç kur|geri sayım)", "timer.start"),
    (r"^\d+ dakika sonra .+ diye haber ver$", "reminder.set"),
    (r"boş muyum|meşgul müyüm", "calendar.list"), (r"takvime ekle|ajandaya ekle", "calendar.add"),
    (r"hakkında (neler bulabilirsin|ne bulabileceğine bak)$", "research.start"),
    (r"hakkımda (ne|neler) biliyorsun$", "memory.recall"), (r"^seninle ne(ler)? yapabilirim$", "help"),
    (r"^çiz: |temalı bir (görsel hazırla|resim çiz)$", "image.generate"),
    (r"^görsele bak: |^(resimdeki|görseldeki) yazıyı oku$", "image.analyze"),
    (r"^anımsat: |^bildir, |^hatırlat, ", "reminder.set"),
]


def test_contrast_pairs_have_correct_labels():
    seen = Counter()
    for cmd, src, args in _contrast():
        datagen.validate_request({"v": 1, "id": "x", "cmd": cmd, "args": args})
        for pat, want in EXPECT:
            if re.search(pat, src):
                assert cmd == want, (src, cmd, want)
                seen[pat] += 1
    assert set(seen) == {p for p, _ in EXPECT}, set(p for p, _ in EXPECT) - set(seen)


def test_contrast_specific_values():
    for cmd, src, args in _contrast(3000, 4):
        if src.startswith("ajandaya ekle, ") or src.startswith("takvime ekle, ") or src.startswith("takvime ekle: "):
            assert src.endswith(" " + args["title"])  # başlık fiilden sonra
        if src.startswith("tercihimi kaydet: "):
            assert args == {"kind": "preference", "text": src[len("tercihimi kaydet: "):]}
        if cmd == "timer.start" and "haber ver diye" in src:
            assert args == {"minutes": int(src.split()[0])}


def test_every_contrast_group_has_two_commands():
    for name, members in datagen.CONTRAST_GROUPS.items():
        assert len({c for c, _ in members}) >= 2, name


def test_contrast_reaches_training_stream():
    srcs = [e["src"] for e in datagen.t2c_set(8000, "train", seed=21)]
    assert any("kayda geç" in s for s in srcs)
    assert any("haber ver diye" in s for s in srcs)
    assert any(re.search(r"(^|\s)çiz: ", s) for s in srcs)


def test_promoted_parts_are_train_and_fresh_parts_held():
    """İstenen söyleyişler eğitim havuzunda; ayrılmış (OOD) yerlerde yeni, hiç eğitilmemiş parçalar var."""
    tr = lambda xs: G.split_pool(xs, "train")  # noqa: E731
    held = lambda xs: G.split_pool(xs, "held")  # noqa: E731
    assert "tercihimi kaydet" in tr(datagen.MEM_PREF) and "alışkanlığımı kaydet" in held(datagen.MEM_PREF)
    assert "anımsat" in tr(datagen.REM_VERBS)
    assert {"{V}: {X}, {T}", "{V}, {T} {X}"} <= set(tr(datagen.REM_FRAMES))
    assert "{V}, {T} {X}" in tr(datagen.CAL_FRAMES) and "{V}: {T} {X}" in tr(datagen.CAL_FRAMES)
    assert {"boş muyum", "meşgul müyüm"} <= set(tr(datagen.CL_PRED))
    assert "seninle neler yapabilirim" in tr(datagen.HELP_PHR)
    assert "görsele bak: {Q}" in tr(datagen.IMGA_Q) and ("resimdeki yazıyı oku", None) in tr(datagen.IMGA)
    assert "{X} temalı bir görsel hazırla" in tr(datagen.IMGG_PRE)
    calc = [f for f, _ in tr(datagen.CALC_FORMS)]
    for w in ("kere", "çarpı", "bölü", "artı", "eksi", "yüzde"):
        assert any(w in f for f in calc), w
    # "unutmamak için kaydet" legacy_ood'da da var: ayrılmış kalır
    assert "unutmamak için kaydet" in held(datagen.NOTE_VERBS) and "bir not yaz" in held(datagen.NOTE_VERBS)
    for pool in (datagen.MEM_PREF, datagen.REM_VERBS, datagen.REM_FRAMES, datagen.CAL_FRAMES, datagen.CL_PRED,
                 datagen.HELP_PHR, datagen.IMGA, datagen.IMGA_Q, datagen.IMGG_PRE, datagen.NOTE_VERBS, datagen.CALC_FORMS):
        assert held(pool) and set(map(str, held(pool))).isdisjoint(map(str, tr(pool)))


def test_contrast_frames_not_held_out():
    """Karşıt çiftler (yalnız eğitim) ayrılmış fiil/yüklem parçalarını içermez → test_ood dürüst kalır."""
    held_phrases = set()
    for pool in (datagen.NOTE_VERBS, datagen.MEM_PREF, datagen.MEM_NOTE, datagen.REM_VERBS, datagen.CL_PRED,
                 datagen.HELP_PHR, datagen.CAL_VERBS, datagen.TIMER_VERBS, datagen.RES_VERBS):
        held_phrases |= set(G.split_pool(pool, "held"))
    held_phrases |= {s for s, _ in G.split_pool(datagen.IMGA, "held")}
    for cmd, src, _ in _contrast(4000, 5):
        for h in held_phrases:
            assert not re.search(r"(^|[\s,:])" + re.escape(h) + r"($|[\s,:])", src), (h, src)


def test_contrast_not_in_legacy_ood_or_fixed_sets():
    # legacy_ood'un ayrılmış eski kalıpları (dilbilgisi üreticisine dayanan komutlarda ayrılmış kalıp yoktur)
    rng = random.Random(202)
    legacy = set()
    for cmd, ts in datagen.TEMPLATES.items():
        if len(ts) > datagen.OOD_N + 2:
            for _ in range(300):
                tr, _a = rng.choice(datagen._templates_for(cmd, "test_ood"))(rng, "test_ood")
                legacy |= {tr, tr.strip()}
    fixed = set()
    for name in ("handwritten-eval.jsonl", "blind-eval.jsonl"):
        fixed |= {json.loads(x)["src"] for x in (SHARED / "core-model" / name).read_text(encoding="utf-8").splitlines() if x.strip()}
    out = _contrast(6000, 6)
    assert not {s for _, s, _ in out} & legacy
    # El yazımı kümede "neler yapabilirsin" gibi argümansız sabit cümleler eskiden beri eğitimde de var;
    # argümanlı hiçbir karşıt cümle sabit kümelerde yok.
    assert not {s for _, s, a in out if a} & fixed


def test_train_stream_never_contains_fixed_eval_sentences():
    """Kör kümenin serbest metinli cümleleri eğitim akışında birebir hiç görülmez.
    El yazımı kümedeki kısa sabit cümleler ("yardım", "ekteki dosyayı özetle") Tur 13'ten beri eğitimde de vardır."""
    free = {"text", "title", "topic", "prompt", "question", "goal", "query", "expr", "need", "label", "name"}
    blind = set()
    for x in (SHARED / "core-model" / "blind-eval.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            if free & set(json.loads(r["tgt"])["args"]):  # serbest metinli cümleler (kısa sabit soru kalıpları hariç)
                blind.add(r["src"])
    assert len(blind) > 10
    srcs = {e["src"] for e in datagen.t2c_set(20000, "train", seed=31)}
    assert not srcs & blind


def test_train_labels_consistent():
    """Aynı cümle eğitimde iki farklı komuta etiketlenmez (model çelişkili veriyle yanlış iş öğrenir)."""
    by_src: dict = {}
    for e in datagen.t2c_set(20000, "train", seed=41):
        by_src.setdefault(e["src"], set()).add(json.loads(e["tgt"])["cmd"])
    bad = {s: c for s, c in by_src.items() if len(c) > 1}
    assert not bad, list(bad.items())[:5]


# --------------------------------------------------------------------------- 3. saat söyleyişleri

_WORD = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
         "on bir": 11, "on iki": 12}
_DOW = dict(datagen.DOW_TR)
_PART = r"(?:(sabah|öğleden sonra|öğlen|öğle|akşam|gece) )?(?:saat )?"
_NUMW = r"(\d{1,2}|on bir|on iki|bir|iki|üç|dört|beş|altı|yedi|sekiz|dokuz|on)"


def _adj(part, h):
    if part in ("öğleden sonra", "akşam") and 1 <= h <= 11:
        return h + 12
    if part == "gece" and 6 <= h <= 11:
        return h + 12
    if part in ("öğle", "öğlen") and 1 <= h <= 5:
        return h + 12
    return h


def _num(s):
    return int(s) if s.isdigit() else _WORD[s]


def _parse_clock(c: str) -> str:
    """Bağımsız başvuru çözümleyici (Kotlin TimeParser yorumu + çeyrek geçe/kala)."""
    for pat, fn in (
        (_PART + r"(\d{1,2})'\w+ çeyrek geçe$", lambda p, n: (_adj(p, int(n)), 15)),
        (_PART + r"(\d{1,2})'\w+ çeyrek kala$", lambda p, n: (_adj(p, int(n)) - 1, 45)),
        (_PART + _NUMW + r" buçukta$", lambda p, n: (_adj(p, _num(n)), 30)),
        (_PART + r"(\d{1,2}):(\d\d)'\w+$", None),
        (_PART + r"(\d{1,2})'(?:de|da|te|ta)$", lambda p, n: (_adj(p, int(n)), 0)),
        (_PART + _NUMW + r"(?:de|da|te|ta)$", lambda p, n: (_adj(p, _num(n)), 0)),
    ):
        m = re.fullmatch(pat, c)
        if m:
            if fn is None:
                h, mm = _adj(m.group(1), int(m.group(2))), int(m.group(3))
            else:
                h, mm = fn(m.group(1), m.group(2))
            return f"{h:02d}:{mm:02d}"
    if c == "öğlen":
        return "12:00"
    raise AssertionError("çözülemedi: " + c)


def _parse_time(tr: str) -> str:
    if tr == "yarım saat sonra":
        return "+30m"
    if tr == "bu gece yarısı":
        return "d+1@00:00"
    m = re.fullmatch(r"(\d+) (dakika|saat|gün) sonra", tr)
    if m:
        return f"+{m.group(1)}" + {"dakika": "m", "saat": "h", "gün": "d"}[m.group(2)]
    m = re.fullmatch(r"(\d+) gün sonra (.+)", tr)
    if m:
        return f"d+{m.group(1)}@" + _parse_clock(m.group(2))
    for w, d in (("yarın ", "d+1"), ("bugün ", "d+0"), ("öbür gün ", "d+2")):
        if tr.startswith(w):
            return f"{d}@" + _parse_clock(tr[len(w):])
    first = tr.split(" ")[0]
    if first in _DOW:
        return f"{_DOW[first]}@" + _parse_clock(tr[len(first) + 1:])
    m = re.fullmatch(r"(\d{1,2}) (\w+?)(?:da|de|ta|te)? (.+)", tr)
    if m and (m.group(2).lower() in datagen.MONTHS_TR):
        mo = datagen.MONTHS_TR.index(m.group(2).lower()) + 1
        return f"{mo:02d}-{int(m.group(1)):02d}@" + _parse_clock(m.group(3))
    return "d+0@" + _parse_clock(tr)


def test_time_phrases_expected_values():
    forms = Counter()
    for split in ("train", "test_iid"):
        rng = random.Random(77)
        for _ in range(20000):
            t = datagen.random_time(rng, split=split)
            assert _parse_time(t.tr) == t.when, (t.tr, t.when)
            for k in ("çeyrek geçe", "çeyrek kala", "buçukta", "öğlen", "gece yarısı", "sabah yedide", "akşam", "kasım"):
                if k in t.tr.lower():
                    forms[(split, k)] += 1
    for k in ("çeyrek geçe", "çeyrek kala", "buçukta", "öğlen", "gece yarısı", "akşam", "kasım"):
        assert forms[("train", k)] > 0, k
    # Kotlin TimeParser'ın çözmediği biçimler yalnızca eğitimde
    assert forms[("test_iid", "çeyrek geçe")] == forms[("test_iid", "gece yarısı")] == 0


def test_word_hours_and_suffixes():
    assert [datagen.word_loc(n) for n in (7, 8, 9, 10, 11, 12, 6, 3)] == \
        ["yedide", "sekizde", "dokuzda", "onda", "on birde", "on ikide", "altıda", "üçte"]
    assert datagen.month_loc("kasım") == "kasımda" and datagen.month_loc("mart") == "martta" \
        and datagen.month_loc("eylül") == "eylülde" and datagen.month_loc("aralık") == "aralıkta"
    rng = random.Random(1)
    seen = {datagen.random_time(rng, split="train").tr for _ in range(30000)}
    assert any(re.search(r"sabah (yedi|sekiz|dokuz)de|sabah dokuzda", s) for s in seen)
    assert any(re.search(r"akşam (altı|yedi|sekiz|dokuz|on) ?\w*(de|da)$", s) for s in seen)
    assert any("akşam" in s and "buçukta" in s for s in seen)
    assert any(re.match(r"\d{1,2} [A-ZÇŞĞÜÖİ]", s) for s in seen)  # "1 Kasım ..."


def test_absolute_date_rate():
    rng = random.Random(5)
    ts = [datagen.random_time(rng, split="test_iid") for _ in range(5000)]
    rate = sum(bool(re.match(r"\d\d-\d\d@", t.when)) for t in ts) / len(ts)
    assert 0.17 < rate < 0.26, rate


# --------------------------------------------------------------------------- 4. hesap

def test_calc_operators_and_numbers():
    rng = random.Random(8)
    words = Counter()
    for _ in range(6000):
        s, args = datagen.g_calc(rng, "train", False, False)
        e = args["expr"]
        nums = re.findall(r"\d+", e)
        assert all(n in s for n in nums if n != "100" or "yüzde" not in s), (s, e)
        for w, op in (("kere", "*"), ("çarpı", "*"), ("bölü", "/"), ("artı", "+"), ("eksi", "-"), ("yüzde", "/100")):
            if w in s:
                assert op in e, (s, e)
                words[w] += 1
        datagen.validate_request({"v": 1, "id": "x", "cmd": "calc.eval", "args": args})
    assert min(words.values()) > 50 and words["kere"] > 300, words
    assert datagen.num_genitive(250) == "250'nin" and datagen.num_possessive(18) == "18'i"
    assert datagen.num_genitive(40) == "40'ın" and datagen.num_possessive(20) == "20'si"


# --------------------------------------------------------------------------- 5. beceri hedefi

def test_skill_goal_keeps_final_verb():
    rng = random.Random(2)
    ends = 0
    n = 0
    for _ in range(3000):
        s, args = datagen.g_skill_learn(rng, "train", False, False)
        assert s.endswith(": " + args["goal"])
        n += 1
        if args["goal"].split()[-1] in {v.split()[-1] for v in datagen.SKILL_FINAL_VERBS}:
            ends += 1
    assert ends / n > 0.5
    assert set(datagen._split(datagen.SKILL_GOALS, "test_iid")).isdisjoint(
        {datagen.g_skill_learn(random.Random(i), "train", False, False)[1]["goal"] for i in range(2000)})
