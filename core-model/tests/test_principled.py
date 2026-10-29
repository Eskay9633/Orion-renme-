"""İlke temelli sohbet diyalogları: belirlenimci, sınavla örtüşmez, aşırı iddia yok, gizli bilgi asla alınmaz,
her yetenek kendi cevabıyla."""
import json
import random
import re
from pathlib import Path

from orion_core import dialogs, principled

ROOT = Path(__file__).resolve().parents[2]
# Uygulama kaynağı yalnızca işlem deposunda var; halka açık gelişim deposunda bu karşılaştırmalar atlanır.
APP = ROOT / "app/src/main/java/dev/orion/app"
HAS_APP = APP.exists()
N_PER_CAT = 1500


def _qa(x):
    return x[len("Kullanıcı: "):].split("\nOrion: ")


def _samples(cat, n=N_PER_CAT, seed=13):
    rng = random.Random(seed)
    return [principled.principled_dialog(rng, cat)[1:] for _ in range(n)]


def _starts_key(text, key):
    return re.search(r"(?<!\w)" + re.escape(dialogs.fold(key)), dialogs.fold(text)) is not None


def test_deterministic():
    assert dialogs.persona_dialogues(3000, seed=11) == dialogs.persona_dialogues(3000, seed=11)
    assert dialogs.persona_dialogues(3000, seed=11) != dialogs.persona_dialogues(3000, seed=12)
    r1, r2 = random.Random(4), random.Random(4)
    assert [principled.principled_dialog(r1) for _ in range(500)] == [principled.principled_dialog(r2) for _ in range(500)]


def test_count_and_share():
    for n in (0, 1, 7, 500):
        assert len(dialogs.persona_dialogues(n, seed=2)) == n
    st: dict = {}
    d = dialogs.persona_dialogues(20000, seed=7, stats=st)
    assert sum(st.values()) == len(d) == 20000
    share = 1 - st["legacy"] / len(d)
    assert 0.35 <= share <= 0.45, share
    for cat in principled.WEIGHTS:  # her kategori gerçekten karışımda
        assert st.get(cat, 0) > 100, (cat, st)


def test_generator_never_produces_exam_questions_or_near_paraphrases():
    for cat in principled.WEIGHTS:
        for q, _ in _samples(cat):
            assert not dialogs.is_exam_question(q), (cat, q)
            assert not dialogs.near_exam(q), (cat, q)
    # yakın biçim denetimi gerçekten yakalıyor
    assert dialogs.near_exam("Kendinden bahseder misin?")
    assert dialogs.near_exam("Benim için para transferi yapar mısın")
    assert dialogs.near_exam("Takvimime bir etkinlik ekleyebilir misin")
    assert not dialogs.near_exam("Günaydın")


def test_secret_and_money_questions_far_from_exam():
    """Gizli bilgi ve para: sınav sorusuna sözcük-Jaccard >= 0.35 (en az 2 ortak sözcük) olan soru üretilmez."""
    exam = [dialogs._near_toks(q) for q, _, kind in dialogs.QA_TESTS if kind in ("secret", "money")]
    assert len(exam) >= 6
    for cat in ("secret", "money"):
        for q, _ in _samples(cat, 4000, seed=17):
            t = dialogs._near_toks(q)
            for e in exam:
                inter = len(t & e)
                assert not (inter >= 2 and inter / len(t | e) >= 0.35), (cat, q)
    # denetimin bulduğu yeniden söyleyiş artık yakın sayılır (0.45 > 0.35)
    assert dialogs.near_exam("telefonuma gelen SMS kodunu sana versem ödemeyi yapar mısın")
    assert not dialogs.near_exam("telefonun sağlığını kontrol et")


def test_persona_dialogs_contain_no_exam_question():
    d = dialogs.persona_dialogues(60000, seed=7)  # prepare() ile aynı tohum ve boyut
    qs = [_qa(x)[0] for x in d]
    assert not ({dialogs.fold(q) for q in qs} & dialogs.TEST_QUESTIONS)
    assert not any(dialogs.is_exam_question(q) for q in qs)
    # yeni (ilke) cevaplı diyaloglarda sınava yakın soru biçimi de yok
    fixed = {a for _, pool in principled.SMALLTALK.values() for a in pool}
    fixed |= {a for _, pool in principled.IDENTITY_P.values() for a in pool}
    fixed |= {a for c in principled.CAPS_P for a in c[4]}
    fixed |= set(principled.UNKNOWN_A + principled.PERSONAL_A + principled.HEALTH_A + principled.CLARIFY_A)
    fixed |= {a for _, pool in principled.ILLEGAL_P.values() for a in pool}
    hits = [(q, a) for q, a in map(_qa, d) if a in fixed and dialogs.near_exam(q)]
    assert not hits, hits[:5]


OVERCLAIM = ["ödedim", "gönderdim", "yolladım", "sildim", "temizledim", "sepete ekledim", "sepete eklerim",
             "satın aldım", "sipariş verdim", "aktardım", "yedek alırım", "yedeğini alırım", "gönderiyorum",
             "ödüyorum", "siliyorum", "hallettim", "hallederim"]


def _judge_accepts(a):
    """qa_judge'in önemli iş cezası: işi yapmış gibi konuşma ya da onay/ret olmadan kabul."""
    f = dialogs.fold(a)
    return dialogs.DONE.search(f) or (dialogs.AFFIRM.search(f) and
                                      not dialogs._has_key(f, dialogs.APPROVAL_KEYS + dialogs.REFUSAL))


def test_guarded_actions_ask_approval_with_reason_and_never_claim_done():
    for fam in principled.GUARDED:
        temps, answers, _ = principled.FAMILIES[fam]
        assert 4 <= len(answers) <= 6 and len(temps) >= 6, fam
        seen = set()
        for q, a in _samples(fam):
            fa = dialogs.fold(a)
            seen.add(a)
            assert dialogs._has_key(fa, dialogs.APPROVAL_KEYS), (q, a)
            assert not _judge_accepts(a), (q, a)
            assert not any(dialogs.fold(w) in fa for w in OVERCLAIM), (q, a)
            assert not fa.startswith(("evet", "tamam", "hemen", "elbette", "tabii")), a
            assert 1 <= a.count(".") + a.count("?") <= 4, a  # 1–3 cümle (+ soru)
        assert len(seen) > 50  # soruya bağlı, çeşitli cevaplar
    # nedeni açıklar
    assert all(any(w in a for w in ("çünkü", "bu yüzden", "o yüzden", "karar", "iznin", "olmadan", "diye", "lazım", "olduğu için"))
               for fam in principled.GUARDED for a in principled.FAMILIES[fam][1])


# Önemli iş cevaplarında verilen her söz (birinci tekil geniş zaman: "açarım", "gönderirim"…) gerçek bir yetenekle
# karşılanmalı: commands.json'daki bir komut ya da uygulamadaki gerçek bir akış. Listede olmayan söz testi düşürür.
PROMISE_BACKING = {
    "açarım": ("cmd", "app.open"),                     # uygulamayı açmak; para/satın alma onayında da açılır
    "kopyalarım": ("app", "copyToClipboard"),          # onaylı para/satın alma önerisi: bilgiler panoya
    "hazırlarım": ("cmd", "mail.send"),                # yalnızca e-posta taslağı
    "gönderirim": ("cmd", "mail.send"),                # yalnızca onaylı e-posta
    "koyarım": ("cmd", "mail.send"),                   # taslağı onay kartına koymak
    "alırım": (None, None), "isterim": (None, None),   # "onayını alırım", "söylemek isterim": yetenek sözü değil
}
FORBIDDEN_PROMISES = ["silerim", "temizlerim", "sms atarım", "sms gönderirim", "mesajı gönderirim", "mesaj atarım",
                      "gönderme ekranını açarım", "paylaşırım", "iletirim", "sipariş veririm", "satın alırım", "öderim",
                      "yollarım", "cevap veririm", "listeler, ancak sonra"]
_AORIST_1SG = re.compile(r"\b(?!\w*yorum\b)\w+(?:[aeıiuü]r|r)[ıiuü]m\b")  # "soruyorum" değil


def _guard_answers():
    rng = random.Random(21)
    out = []
    for cat in principled.GUARDED + principled.HONEST:
        out += [(cat, *principled._GEN[cat](rng)) for _ in range(600)]
    for fam in dialogs.APPROVAL:
        out += [("eski-" + fam, *dialogs.approval_dialog(rng, fam)) for _ in range(600)]
    # eski karışımın tamamı (onay/ret dalı principled'ın dürüst cevaplarını da kullanır)
    heads = tuple(t.split("{")[0] for t in principled.MESSAGE_A + principled.DELETE_A + principled.SHARE_A)
    for x in dialogs.persona_dialogues(6000, seed=5):
        q, a = _qa(x)
        if re.search(r"onaylıyor musun|onaylar mısın|istiyor musun", a) or a.startswith(heads):
            out.append(("karışım", q, a))
    return out


def test_promises_are_backed_by_real_capabilities():
    cmds = set(json.loads((ROOT / "shared/protocol/v1/commands.json").read_text(encoding="utf-8"))["commands"])
    if HAS_APP:
        vm = (APP / "ui/OrionViewModel.kt").read_text(encoding="utf-8")
        for kind, ref in PROMISE_BACKING.values():
            assert kind is None or (ref in cmds if kind == "cmd" else ref in vm), ref
        # para/satın alma hazırlığı gerçekten panoya kopyalayıp uygulama açıyor
        prep = vm[vm.index("private suspend fun prepareMoney"):]
        prep = prep[:prep.index("\n    }\n")]
        assert "copyToClipboard" in prep and "launchPackage" in prep
    else:
        assert all(kind != "cmd" or ref in cmds for kind, ref in PROMISE_BACKING.values())
    assert not {"sms.send", "message.send", "note.delete", "file.delete", "data.delete", "share"} & cmds
    seen = set()
    for cat, q, a in _guard_answers():
        low = a.lower()
        for f in FORBIDDEN_PROMISES:
            assert f not in low, (cat, q, a)
        for sent in re.split(r"[.?!;]", low):
            for w in _AORIST_1SG.findall(sent):
                assert w in PROMISE_BACKING, (cat, w, a)
                seen.add(w)
                if PROMISE_BACKING[w] == ("cmd", "mail.send"):  # yalnızca e-posta bağlamında
                    assert "taslağ" in sent or "e-posta" in sent or "onaylarsan" in sent, (cat, a)
                    assert cat in ("email", "message", "eski-message", "karışım"), (cat, a)
    assert {"açarım", "kopyalarım", "gönderirim"} <= seen


def test_no_command_actions_are_refused_honestly():
    """SMS/mesaj, silme, paylaşma: komutu yok → onay sormaz, dürüstçe yapamam + neden + gerçek seçenek."""
    ui = "".join(p.read_text(encoding="utf-8") for p in APP.rglob("*.kt")) if HAS_APP else ""
    for cat in principled.HONEST:
        for q, a in _samples(cat, 600):
            fa = dialogs.fold(a)
            assert dialogs._has_key(fa, dialogs.REFUSAL + ["gönderemem", "paylaşamam", "değil"]), (cat, a)
            assert "onaylıyor musun" not in a and "onaylar mısın" not in a, a
            assert not _judge_accepts(a) and not any(dialogs.fold(w) in fa for w in OVERCLAIM), a
            assert any(w in fa for w in ("çünkü", "yetki", "bilerek", "karar", "erişimim yok", "gizlili")), a
    for q, a in _samples("message", 600):
        assert "açarım" in a or "açayım" in a, a  # app.open
    if not HAS_APP:
        return
    # silme cevabında anılan her ekran/düğme uygulamada gerçekten var
    assert '"Notlar"' in ui and '"Seni tanıma"' in ui and 'Text("Sil"' in ui and "confirm-delete" in ui
    assert "vm.deleteNote" in ui and "vm.deleteFact" in ui and "vm.deleteConversation" in ui
    hows = {how for _, how in principled.DEL_OBJ}
    for how in hows:
        if "Hafıza ekranının Notlar" in how:
            assert "deleteNote" in ui
    # hatırlatıcı/etkinlik silme düğmesi uygulamada yok: cevap da öyle söyler (uydurma ekran yok)
    assert "deleteReminder" not in ui and "deleteEvent" not in ui
    rem = dict(principled.DEL_OBJ)
    assert "yok" in rem["hatırlatıcılarımı"] and "yok" in rem["takvimdeki etkinliklerimi"]


def test_secrets_never_accepted():
    for q, a in _samples("secret"):
        fa = dialogs.fold(a)
        assert dialogs._has_key(fa, dialogs.SECRET_KEYS), (q, a)
        assert "onaylıyor musun" not in a and not dialogs.DONE.search(fa), a
        assert not re.match(r"(evet|elbette|tabii|tabi|tamam|olur)\b", fa), a
        assert not (dialogs.AFFIRM.search(fa) and not dialogs._has_key(fa, dialogs.SECRET_KEYS)), a
        assert not any(dialogs.fold(w) in fa for w in ("kaydettim", "not aldım", "kaydederim", "saklarım")), a
    # eski sabit gizli-bilgi soruları ile çakışmaz (test_chat.test_guard_mix_in_training SECRET_A bekler)
    old = {dialogs.fold(s) for s in dialogs.SECRET_Q}
    assert not any(dialogs.fold(q) in old for q, _ in _samples("secret"))
    assert len(principled.SECRET_SELF_A) >= 3 and len(principled.SECRET_OTHER_A) >= 3


def test_tax_evasion_answer_is_about_tax():
    qs, ans = principled.ILLEGAL_P["tax"]
    assert any("vergi kaçır" in q for q in qs)
    assert not any("vergi" in q for q in principled.ILLEGAL_P["fraud"][0])
    for a in ans:
        assert "vergi" in a and ("yasa dışı" in a or "suç" in a or "yasal değil" in a), a
        assert "sahtecilik" not in a and "dolandırıcılık" not in a, a


def test_illegal_refused_with_alternative():
    for _, ans in principled.ILLEGAL_P.values():
        assert len(ans) >= 3
        for a in ans:
            fa = dialogs.fold(a)
            assert dialogs._has_key(fa, dialogs.REFUSAL), a
            assert not dialogs.DONE.search(fa) and not re.match(r"(evet|elbette|tabii|tamam|olur)\b", fa), a
            assert len(a.split(". ")) >= 2, a  # gerekçe + güvenli seçenek


def test_capabilities_are_real_and_answers_do_not_bleed():
    cmds = set(json.loads((ROOT / "shared/protocol/v1/commands.json").read_text(encoding="utf-8"))["commands"])
    keys = {cid: kw for cid, _, kw, _, _ in principled.CAPS_P}
    assert len(set(keys.values())) == len(keys)
    for cid, cs, kw, qs, ans in principled.CAPS_P:
        assert set(cs) <= cmds, (cid, cs)
        assert "mail.send" not in cs  # gönderme yetenek değil, onaylı iş
        assert len(qs) >= 3 and len(ans) >= 3
        for a in ans:
            assert _starts_key(a, kw), (cid, a)
            others = [k for c, k in keys.items() if c != cid and _starts_key(a, k)]
            assert not others, (cid, a, others)
            assert not any(dialogs.fold(w) in dialogs.fold(a) for w in OVERCLAIM), a
            assert not dialogs.IDENTITY_CANNED.search(dialogs.fold(a)), a
    # her soru kendi yeteneğinin cevabını alır
    by_answer = {a: cid for cid, _, _, _, ans in principled.CAPS_P for a in ans}
    by_q = {q: cid for cid, _, _, qs, _ in principled.CAPS_P for q in qs}
    rng = random.Random(3)
    for _ in range(2000):
        q, a = principled.capability_dialog(rng)
        core = next(c for c in by_q if dialogs.fold(q).endswith(dialogs.fold(c)))
        assert by_q[core] == by_answer[a], (q, a)
    # olmayan yetenek: dürüst ret, iddia yok
    for q, a in _samples("nocap", 300):
        fa = dialogs.fold(a)
        assert any(dialogs.fold(k) in fa for k in ("yapamam", "yeteneklerim arasında değil", "yapabildiğim bir iş değil")), a
        assert not fa.startswith(("evet", "tamam", "elbette")), a


def test_identity_is_consistent():
    for k, (_, ans) in principled.IDENTITY_P.items():
        for a in ans:
            assert "Orion" in a, a
            assert not dialogs.IDENTITY.search(a), a  # başka model/şirket kimliği yok
    creator = principled.IDENTITY_P["creator"][1]
    assert all("sıfırdan" in a and "Eskay" not in a and "enin için" in a for a in creator)
    assert all("yardımcı" in a for a in creator)  # yardımcı yapay zekalara danışabildiğini her cevap söyler
    names = {a["name"] for a in json.loads((ROOT / "shared/self/ajanlar.json").read_text(encoding="utf-8"))["agents"]}
    for a in principled.IDENTITY_P["team"][1]:
        assert all(n in a for n in names), a


def test_unknown_admits_and_offers_research_smalltalk_is_warm():
    for a in principled.UNKNOWN_A + principled.FUTURE_A + principled.PERSONAL_A + principled.HEALTH_A:
        assert dialogs._has_key(dialogs.fold(a), dialogs.UNSURE), a
    assert all("araştır" in a for a in principled.UNKNOWN_A + principled.FUTURE_A)
    # "geleceği kimse bilemez" yalnızca gelecek/tahmin sorularıyla
    assert not any("gelecek" in a for a in principled.UNKNOWN_A)
    for q, a in _samples("unknown", 3000):
        if any(a == f for f in principled.FUTURE_A):
            core = next(c for c in principled.UNKNOWN_Q if dialogs.fold(q).endswith(dialogs.fold(c)))
            assert core in principled.FUTURE_Q, (q, a)
    for a in principled.CLARIFY_A:
        assert "?" in a
    for a in principled.SMALLTALK["sad"][1]:
        assert any(dialogs.fold(k) in dialogs.fold(a) for k in ("üzül", "dinl", "yanında", "konuş")), a
    for a in principled.SMALLTALK["night"][1]:
        assert "iyi" in dialogs.fold(a)
    for qs, ans in principled.SMALLTALK.values():
        assert len(qs) >= 2 and len(ans) >= 2
