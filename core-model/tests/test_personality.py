"""Tur 21 kişilik verisi: kişilik / hafıza notu / akıl yürütme üreticileri, MKQA yükleyicisi, karışım payları.
Ağ kullanılmaz (MKQA yerel küçük fikstürle sınanır)."""
import gzip
import hashlib
import json
import random
import re

import pytest

from orion_core import chat, cogexam, dialogs, memory_notes as M, mkqa, personality as P, reasoning as R

CORPORATE = ["onayınız gerekmektedir", "politikamız", "işleminiz", "değerli kullanıcı",
             "size yardımcı olmaktan memnuniyet duyarım", "memnuniyet duyarım", "talebiniz", "hizmetinizde",
             "sayın kullanıcı", "lütfen bekleyiniz", "değerli müşterimiz", "bu tür işlemler için", "gerekmektedir",
             "yardımcı olabileceğim başka bir konu", "anlayışınız için"]
OVERCLAIM = ["ödedim", "gönderdim", "yolladım", "sildim", "temizledim", "sepete ekledim", "satın aldım", "sipariş verdim",
             "aktardım", "hallettim", "kaydettim", "not aldım", "hatırlatıcı kurdum", "alarm kurdum", "takvime ekledim",
             "silerim", "öderim", "yollarım", "satın alırım", "sipariş veririm", "paylaşırım"]
FEELING_CLAIMS = ["hissediyorum", "kalbim", "duygularım var", "canım yandı", "seni seviyorum", "gözyaşı"]
N = 20000


def _split(x):
    """Belge → (hafıza satırı ya da None, [(kullanıcı, Orion), ...])."""
    line = None
    if x.startswith("Hafıza: "):
        line, x = x.split("\n", 1)
    return line, [tuple(t.split("\nOrion: ")) for t in x[len("Kullanıcı: "):].split("\nKullanıcı: ")]


@pytest.fixture(scope="module")
def mix():
    st: dict = {}
    d = dialogs.persona_dialogues(N, seed=7, stats=st)
    return d, st


def _gen_samples(n=3000, seed=17):
    rng = random.Random(seed)
    out = []
    for cat in P.WEIGHTS:
        out += [("kisilik:" + cat, P.personality_turns(rng, cat)[1]) for _ in range(n // len(P.WEIGHTS))]
    for cat in R.WEIGHTS:
        out += [("akil:" + cat, R.reasoning_turns(rng, cat)[1]) for _ in range(n // len(R.WEIGHTS))]
    return out


# ------------------------------------------------------------------ biçim
def test_memory_prompt_format_exact():
    assert chat.prompt_for("Sen kimsin?") == "Kullanıcı: Sen kimsin?\nOrion:"  # hafızasız: eski biçim aynen
    assert chat.prompt_for("Sen kimsin?", None) == chat.prompt_for("Sen kimsin?", "") == \
        chat.prompt_for("Sen kimsin?", "   ") == "Kullanıcı: Sen kimsin?\nOrion:"
    p = chat.prompt_for("Kızım için ne yazayım?", "kızının adı Ela; 34 yaşındasın")
    assert p == "Hafıza: kızının adı Ela; 34 yaşındasın\nKullanıcı: Kızım için ne yazayım?\nOrion:"
    assert p == p.rstrip() and chat.MEMORY_PREFIX == M.PREFIX
    # eğitim belgesi = istem + " " + cevap (hafıza satırı dahil)
    rng = random.Random(1)
    for _ in range(300):
        d = M.memory_dialog(rng)
        doc = d.line + "\n" + "\n".join(dialogs.format_dialog(q, a) for q, a in d.turns)
        q, a = d.turns[0]
        first = d.line + "\n" + dialogs.format_dialog(q, a)
        assert first == chat.prompt_for(q, d.line[len(M.PREFIX):]) + " " + a
        assert doc.startswith(first)


def test_memory_line_rules():
    """Uygulamayla ortak sözleşme: önek dahil <= 160, <= 3 not, not <= 70, ';' ve sonda '.' yok, küçük harf başlangıç
    (özel ad hariç), notlar yarıda kesilmez."""
    rng = random.Random(3)
    seen_lens = []
    for _ in range(5000):
        d = M.memory_dialog(rng)
        assert M.validate_line(d.line, M.PROPER) == [], d.line
        notes = d.line[len(M.PREFIX):].split("; ")
        assert 1 <= len(notes) <= 3 and len(d.line) <= 160 and all(len(n) <= 70 for n in notes)
        assert {f.note for f in d.facts} == set(notes)  # her not bütün bir gerçek (kesilmemiş)
        seen_lens.append(len(notes))
    assert {1, 2, 3} <= set(seen_lens)
    # sığmayan not bütünüyle düşer; geçersiz not alınmaz
    long = ["a" * 70, "b" * 70, "c" * 70]
    line = M.memory_line(long)
    assert line == "Hafıza: " + "a" * 70 + "; " + "b" * 70 and len(line) <= 160
    assert M.memory_line(["x;y", "sonu nokta."]) == ""
    assert M.validate_line("Hafıza: Kahveyi sever", M.PROPER) and M.validate_line("Hafıza: a; b; c; d")
    assert M.validate_line("Hafıza: Konya'da yaşıyorsun; İspanyolca öğreniyorsun", M.PROPER) == []


def test_memory_lines_in_training_mix_valid(mix):
    d, _ = mix
    lines = [x.split("\n", 1)[0] for x in d if x.startswith("Hafıza: ")]
    assert len(lines) > 0.12 * N
    assert all(M.validate_line(l, M.PROPER) == [] for l in lines)


def _values_in(text):
    f = f" {dialogs.fold(text)} "
    return {v for v in M.ALL_VALUES if f" {dialogs.fold(v)} " in f}


def test_memory_answers_use_only_note_facts():
    rng = random.Random(5)
    n = {t: 0 for t in M.TYPES}
    for _ in range(6000):
        d = M.memory_dialog(rng)
        n[d.cat] += 1
        line_vals = _values_in(d.line)
        answers = " ".join(a for _, a in d.turns)
        if d.cat in ("kullan", "hatirla"):
            assert _values_in(answers) <= line_vals, (d.line, d.turns)  # notta olmayan ad/tarih/şehir uydurulmaz
            if d.used.values:  # kullanılan notun değeri cevapta geçer
                low = answers.lower()
                assert any(v.lower() in low for v in d.used.values), (d.line, d.turns)
        else:
            assert not (_values_in(answers) & line_vals), (d.line, d.turns)  # ilgisiz notlar anılmaz
        if d.cat == "yok":
            a = d.turns[0][1]
            assert "bilmiyorum" in a.lower() or "not yok" in a or "yok" in a, a
            assert "?" in a, a  # uydurmak yerine sorar
            assert not _values_in(a), a
    assert all(v > 500 for v in n.values()), n


def test_memory_names_fictional_and_not_from_exams():
    exam_toks = {w for it in cogexam.load_items() for t in it["turns"] for w in dialogs.fold(t).split()}
    exam_toks |= {w for q, *_ in dialogs.QA_TESTS for w in dialogs.fold(q).split()}
    names = set(M.KID_NAMES) | set(M.PARTNER_NAMES) | set(M.PET_NAMES) | {c for c, _ in M.CITIES}
    clash = {n for n in names if dialogs.fold(n) in exam_toks}
    assert not clash, clash
    legacy = {dialogs.fold(n[0].split("'")[0]) for n in dialogs.NAMES}
    assert not {dialogs.fold(n) for n in names} & legacy  # eski para/mesaj diyaloglarının adları da değil


# ------------------------------------------------------------------ kişilik
def test_no_corporate_voice_anywhere(mix):
    d, _ = mix
    pools = [a for _, ts in _gen_samples() for _, a in ts]
    pools += [a for x in d for _, a in _split(x)[1]]
    for a in pools:
        low = a.lower()
        assert not any(c in low for c in CORPORATE), a


def test_efendim_rare_and_never_in_serious_answers(mix):
    d, st = mix
    pers = []
    for cat, ts in _gen_samples(6000, seed=23):
        if cat.startswith("kisilik:"):
            pers += [(cat, a) for _, a in ts]
    with_e = [a for _, a in pers if "efendim" in a.lower()]
    assert 0 < len(with_e) / len(pers) <= 0.2, len(with_e) / len(pers)
    assert all(a.lower().count("efendim") == 1 for a in with_e)
    assert not any("efendim" in a.lower() for c, a in pers if c == "kisilik:moral")
    # para/onay/gizli bilgi cevaplarında asla
    for fam in ("money", "purchase", "email", "secret", "illegal"):
        rng = random.Random(2)
        for _ in range(300):
            assert not any("efendim" in a.lower() for _, a in dialogs_principled(rng, fam))
    # karışımın tamamında da 1/5'i geçmez
    all_a = [a for x in d for _, a in _split(x)[1]]
    assert sum("efendim" in a.lower() for a in all_a) / len(all_a) <= 0.2


def dialogs_principled(rng, fam):
    from orion_core import principled
    return principled.principled_turns(rng, fam)[1]


def test_new_answers_no_overclaim_no_identity_no_feelings():
    rng = random.Random(9)
    answers = [a for _, ts in _gen_samples() for _, a in ts]
    answers += [a for _ in range(3000) for _, a in M.memory_dialog(rng).turns]
    for a in answers:
        fa = dialogs.fold(a)
        assert not any(re.search(r"\b" + dialogs.fold(w) + r"\b", fa) for w in OVERCLAIM), a
        assert not dialogs.IDENTITY.search(a), a
        assert not any(w in a.lower() for w in FEELING_CLAIMS), a


def test_personality_varied_and_tone_matched():
    rng = random.Random(4)
    for cat in P.WEIGHTS:
        answers = {a for _ in range(1500) for _, a in P.personality_turns(rng, cat)[1]}
        assert len(answers) > 25, (cat, len(answers))
    # ton: kötü gün → sakin (açılış BAD_OPEN, espri yok); iyi gün → neşeli
    for _ in range(2000):
        t = P.day_turns(rng)
        u, a = t[1]
        if any(u.startswith(l.split(",")[0]) for l in P.BAD_LEAD):
            assert any(a.startswith(o) for o in P.BAD_OPEN), a
            assert not any(j in a for j in P.JOKES)
        if any(u.startswith(l.split(",")[0].split("!")[0]) for l in P.GOOD_LEAD):
            assert any(a.startswith(o) for o in P.GOOD_OPEN), a
    for _ in range(500):
        a = P.down_turns(rng)[0][1]
        assert "?" in a and not any(j in a for j in P.JOKES)


def test_opinions_consistent_and_deferential():
    rng = random.Random(6)
    picks = {}
    for _ in range(3000):
        q, a = P.opinion_turns(rng)[0]
        for x, _, y, _ in P.OPINIONS:
            if f" {x} " in f" {q.lower()} " and f" {y} " in f" {q.lower()} ":
                picks.setdefault(x, set()).add(next(p for p in P.OP_PICK if p.format(x=x) in a))
                assert any(dd in a for dd in P.OP_DEFER), a  # son söz sahibinde
                assert a.find(x) < (a.find(y) if y in a else len(a)), a  # kendi tercihi önce
    assert len(picks) >= len(P.OPINIONS) - 2


def test_satisfaction_moments():
    rng = random.Random(8)
    pos = neg = 0
    for _ in range(2000):
        t = P.satisfaction_turns(rng)
        u, a = t[1]
        if u in P.POS:
            pos += 1
            assert a in P.POS_A and ("sevin" in a.lower() or "memnun" in a.lower()), a
        else:
            neg += 1
            assert u in P.NEG and a in P.NEG_A and "?" in a
            if len(t) == 3:
                assert t[2][1] != t[0][1]  # yeni bir deneme
    assert pos > 600 and neg > 600
    assert any(P.satisfaction_turns(random.Random(i))[1][0] == "😊" for i in range(200))
    assert any(P.satisfaction_turns(random.Random(i))[1][0] == "🙁" for i in range(200))


def test_identity_from_orion_json():
    import json as _j
    from orion_core import SHARED
    me = _j.loads((SHARED / "self" / "orion.json").read_text(encoding="utf-8"))
    assert P.NAME == me["name"] == "Orion" and P.HONORIFIC == me["address"]["honorific"]
    assert P.EFENDIM_P <= 1 / me["address"]["every"]


# ------------------------------------------------------------------ akıl yürütme
_EQ = re.compile(r"(\d+) ([×÷+−]) (\d+)(?: \+ (\d+))? = (\d+)")


def test_reasoning_arithmetic_is_correct():
    rng = random.Random(11)
    n = 0
    for _ in range(4000):
        cat, ts = R.reasoning_turns(rng)
        for _, a in ts:
            for m in _EQ.finditer(a):
                x, op, y, z, r = m.groups()
                x, y, r = int(x), int(y), int(r)
                v = {"×": x * y, "÷": x // y, "+": x + y + int(z or 0), "−": x - y}[op]
                assert v == r, a
                n += 1
    assert n > 1000


def test_reasoning_contradiction_asks_and_rules_applied():
    rng = random.Random(12)
    for _ in range(500):
        a = R.contradiction_turns(rng)[-1][1]
        assert "?" in a and ("önce" in a or "az önce" in a), a
    for _ in range(300):
        t = R.context3_turns(rng)
        assert len(t) == 3


def test_new_generators_exam_disjoint(mix):
    """Karışımdaki her kullanıcı turu sınav dışı: QA_TESTS (aynısı) ve bilişsel sınav (yakını); hafıza notlu
    belgelerde QA_TESTS'e yakın biçimler de yok. Yeni üreticilerin süzgeçten geçemeyen oranı küçük."""
    d, _ = mix
    for x in d:
        line, ts = _split(x)
        for q, _ in ts:
            assert not dialogs.is_exam_question(q) and not cogexam.near_cogexam(q), q
            if line:
                assert not dialogs.near_exam(q), q
    rng = random.Random(13)
    assert sum(not dialogs._turns_ok(M.memory_dialog(rng).turns) for _ in range(2000)) < 100
    samples = _gen_samples(2000, seed=31)
    assert sum(not dialogs._turns_ok(ts) for _, ts in samples) < 0.1 * len(samples)


# ------------------------------------------------------------------ karışım
def test_mix_shares_within_tolerance(mix):
    _, st = mix
    sh = dialogs.mix_shares(st)
    for k, v in dialogs.MIX.items():
        assert abs(sh[k] - v) < 0.02, (k, sh)
    assert abs(sum(dialogs.MIX.values()) - 1) < 1e-9
    assert dialogs.PRINCIPLED_SHARE == pytest.approx(0.15)


def test_determinism():
    assert dialogs.persona_dialogues(2000, seed=3) == dialogs.persona_dialogues(2000, seed=3)
    r1, r2 = random.Random(5), random.Random(5)
    assert [P.personality_turns(r1) for _ in range(300)] == [P.personality_turns(r2) for _ in range(300)]
    r1, r2 = random.Random(5), random.Random(5)
    assert [str(M.memory_dialog(r1)) for _ in range(300)] == [str(M.memory_dialog(r2)) for _ in range(300)]
    r1, r2 = random.Random(5), random.Random(5)
    assert [R.reasoning_turns(r1) for _ in range(300)] == [R.reasoning_turns(r2) for _ in range(300)]


# ------------------------------------------------------------------ MKQA (yerel fikstür, ağ yok)
def _mkqa_fixture(path, n=60):
    rows = []
    for i in range(n):
        rows.append({"example_id": i, "query": f"q{i}", "queries": {"tr": f"Kurgusal soru numarası {i} kimdir"},
                     "answers": {"tr": [{"type": "entity", "text": f"Kişi {i}", "aliases": [f"K{i}"]}]}})
    rows += [
        {"example_id": 1000, "queries": {"tr": "bir piramidin kaç köşesi vardır"},
         "answers": {"tr": [{"type": "number", "text": "5.0", "aliases": ["5"]}]}},
        {"example_id": 1001, "queries": {"tr": "özgürlük bildirgesi ne zaman imzalandı"},
         "answers": {"tr": [{"type": "date", "text": "1776-07-04"}]}},
        {"example_id": 1002, "queries": {"tr": "denizatı erkekleri doğurur mu"},
         "answers": {"tr": [{"type": "binary", "text": "yes"}]}},
        {"example_id": 1003, "queries": {"tr": "neden gökyüzü mavidir uzun anlat"},
         "answers": {"tr": [{"type": "long_answer", "text": None}]}},
        {"example_id": 1004, "queries": {"tr": "yarın hangi takım kazanacak bilinmez"},
         "answers": {"tr": [{"type": "unanswerable", "text": None}]}},
        {"example_id": 1005, "queries": {"tr": "Türkiye'nin başkenti neresidir?"},  # sınav sorusu: atılmalı
         "answers": {"tr": [{"type": "entity", "text": "Ankara"}]}},
        {"example_id": 1006, "queries": {"tr": "yük vagonu ne kadar uzundur"},
         "answers": {"tr": [{"type": "number_with_unit", "text": "15.5 metre"}]}},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mkqa_download_checks_hash_and_fails_loudly(tmp_path, monkeypatch):
    src = tmp_path / "src.jsonl.gz"
    sha = _mkqa_fixture(src)
    p = mkqa.download(tmp_path / "c1", str(src), sha)
    assert p.exists() and mkqa._sha(p) == sha
    with pytest.raises(ValueError):
        mkqa.download(tmp_path / "c2", str(src), "0" * 64)

    def no_net(*a, **k):
        raise OSError("ağ yok")
    monkeypatch.setattr(mkqa.urllib.request, "urlretrieve", no_net)
    with pytest.raises(OSError):
        mkqa.download(tmp_path / "c3")  # sabit adres, ağ yok → hata (sessiz atlama yok)
    assert mkqa.MKQA_URL.startswith("https://raw.githubusercontent.com/apple/ml-mkqa/") and mkqa.MKQA_COMMIT in mkqa.MKQA_URL
    assert len(mkqa.MKQA_SHA256) == 64


def test_mkqa_build_conversion_and_holdout(tmp_path):
    src = tmp_path / "m.jsonl.gz"
    _mkqa_fixture(src)
    texts, ev, st = mkqa.build(src, n_eval=20)
    assert len(ev) == 20 and st["degerlendirme"] == 20
    assert st["atilan"].get("sinav_ya_da_suzgec", 0) == 1 and st["atilan"].get("uzun_ya_da_bos", 0) == 1
    qs_train = {dialogs.fold(t.split("\nOrion: ")[0][len("Kullanıcı: "):]) for t in texts}
    assert not qs_train & {dialogs.fold(r["q"]) for r in ev}  # değerlendirme eğitimden ayrı
    assert not any("başkenti" in t for t in texts)
    blob = "\n".join(texts)
    assert texts == mkqa.build(src, n_eval=20)[0]  # belirlenimci
    for t in texts:
        assert t.startswith("Kullanıcı: ") and "\nOrion: " in t
        assert not dialogs.IDENTITY.search(t)
    # değer biçimleri (hiçbiri uydurulmaz)
    pairs = dict(tuple(t[len("Kullanıcı: "):].split("\nOrion: ")) for t in texts)
    a = lambda key: next((v for k, v in pairs.items() if key in k.lower()), None)
    if a("köşesi"):
        assert re.search(r"\b5\b", a("köşesi")) and "5.0" not in a("köşesi")
    if a("bildirgesi"):
        assert "4 Temmuz 1776" in a("bildirgesi")
    if a("denizatı"):
        assert "evet" in a("denizatı").lower()
    if a("vagonu"):
        assert "15,5 metre" in a("vagonu")
    if a("takım kazanacak"):
        assert a("takım kazanacak") in mkqa.UNSURE_T
    assert "Kişi" in blob
    assert mkqa.eval_score(["Kişi 3 olabilir.", "bilmiyorum"], [{"type": "entity", "golds": ["Kişi 3"]},
                                                                {"type": "entity", "golds": ["Kişi 4"]}]) == 0.5


def test_prepare_adds_mkqa_and_writes_eval(tmp_path, monkeypatch):
    src = tmp_path / "m.jsonl.gz"
    sha = _mkqa_fixture(src, n=1100)
    monkeypatch.setattr(mkqa, "MKQA_SHA256", sha)
    monkeypatch.setattr(chat, "persona_dialogues", lambda n, stats=None: ["Kullanıcı: Merhaba\nOrion: Selam!"] * 20)
    texts = [f"Orion telefonda yaşar. Bugün {i}. gün ve hava güzel." * 3 for i in range(60)]
    st = chat.prepare(tmp_path / "d", 10**9, vocab=300, texts=texts, open_data=False, mkqa_src=str(src))
    assert st["mkqa"]["egitim"] > 0 and st["mkqa"]["degerlendirme"] == mkqa.N_EVAL
    assert "MKQA" in st["source"] and "CC BY-SA 3.0" in st["source"]
    ev = (tmp_path / "d" / "mkqa-eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ev) == mkqa.N_EVAL
    # open_data kapalı ve mkqa verilmezse MKQA yok (eski davranış)
    st2 = chat.prepare(tmp_path / "d2", 10**9, vocab=300, texts=texts, open_data=False)
    assert "mkqa" not in st2
