"""MASSIVE 1.1 (tr-TR) / xSID 0.6 (tr) çevirisi: ağsız, yerel küçük örnek dosyalarla (tests/fixtures; Orion için elle
yazılmış, MASSIVE/xSID biçiminde sahte satırlar). Gerçek veri yalnızca ORION_EXT_DATA hazırlanmışsa denetlenir."""
import hashlib
import io
import json
import os
import random
import tarfile
from pathlib import Path

import pytest

from orion_core import SHARED, massive as M
from orion_core.protocol import validate_request

FIX = Path(__file__).parent / "fixtures"


def _tar(tmp_path: Path) -> tuple[Path, str]:
    data = (FIX / "massive-mini.jsonl").read_bytes()
    p = tmp_path / "m.tgz"
    with tarfile.open(p, "w:gz") as tf:
        ti = tarfile.TarInfo(M.MASSIVE_MEMBER)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _xsid(tmp_path: Path) -> tuple[Path, dict]:
    d = tmp_path / "xsid"
    d.mkdir()
    sha = {}
    for split in ("valid", "test"):
        b = (FIX / f"xsid-mini.{split}.conll").read_bytes()
        (d / f"tr.{split}.conll").write_bytes(b)
        sha[f"tr.{split}.conll"] = hashlib.sha256(b).hexdigest()
    return d, sha


@pytest.fixture()
def prepared(tmp_path):
    tar, sha = _tar(tmp_path)
    xd, xsha = _xsid(tmp_path)
    out = tmp_path / "ext"
    stats = M.prepare(out, tar, xd, massive_sha=sha, xsid_sha=xsha)
    return out, stats


def test_when_parser():
    cases = {
        "yarın sabah yedide": "d+1@07:00", "cuma akşam sekize": "fri@20:00", "bugün öğleden sonra üç buçukta": "d+0@15:30",
        "yarın 17:45'te": "d+1@17:45", "on iki mart saat onda": "03-12@10:00", "on dakika sonra": "+10m",
        "iki saat sonra": "+2h", "yarım saat sonra": "+30m", "öğlen": "d+0@12:00", "saat dokuzda": "d+0@09:00",
        "gelecek salı sabah sekiz kırk beşte": "tue@08:45", "yarınki": None, "yarın": None,
    }
    for phrase, want in cases.items():
        got = M.parse_when(phrase)
        assert (got[0] if got else None) == want, (phrase, got)
    # belirsiz ya da çelişkili saat: atla
    for phrase in ("saat beşte", "yarın akşam", "sabah on yedide", "bu hafta", "gece ikide", "akşam on ikide"):
        assert M.parse_when(phrase, default_hour=9) is None, phrase
    # gün var saat yok: hatırlatıcıda Kotlin TimeParser varsayılanı 09:00
    assert M.parse_when("yarınki", default_hour=9) == ("d+1@09:00", None)
    assert M.parse_when("her gün sabah altıda", default_hour=9) == ("d+0@06:00", "daily")
    assert M.parse_day("yarın") == "d+1" and M.parse_day("üç gün sonra") == "d+3" and M.parse_day("cuma") is None


def test_wake_word_and_polite_strip():
    assert M.strip_wake("olly salondaki ışıkları kapat") == "salondaki ışıkları kapat"
    assert M.strip_wake("alexa, sesi aç olly") == "sesi aç"
    assert M.strip_wake("hollywood'daki filmler") == "hollywood'daki filmler"
    assert M.strip_polite("lütfen ışıkları kapat lütfen.") == "ışıkları kapat"
    assert M.norm_key("Olly, Hava Nasıl Olacak?") == M.norm_key("hava nasıl olacak")


def test_quality_filter_majority():
    j = lambda i, g, l="target": {"intent_score": i, "slots_score": 1, "grammar_score": g, "spelling_score": 2,
                                  "language_identification": l}
    assert M.quality_ok([j(1, 4), j(1, 3), j(0, 1)])
    assert not M.quality_ok([j(1, 4), j(0, 4), j(2, 4)])
    assert not M.quality_ok([j(1, 2), j(1, 2), j(1, 4)])
    assert M.quality_ok([j(1, 4, "target|english")] * 3)
    assert not M.quality_ok([j(1, 4, "english")] * 3)
    assert not M.quality_ok([])


def test_conversion_rules(prepared):
    out, stats = prepared
    rows = {r["src"]: r for f in (M.TRAIN_FILE, *M.EVAL_FILES.values()) for r in M._read(out / f)}
    want = {
        "yarın sabah yedide ilaç içmeyi hatırlat": {"cmd": "reminder.set", "args": {"at": "d+1@07:00", "text": "ilaç içmek"}},
        "cuma akşam sekize sinema ekle": {"cmd": "calendar.add", "args": {"start": "fri@20:00", "title": "sinema"}},
        "yarın dişçi randevumu hatırlat": {"cmd": "reminder.set", "args": {"at": "d+1@09:00", "text": "dişçi randevumu"}},
        "on dakika sonra beni uyar": {"cmd": "timer.start", "args": {"minutes": 10}},
        "kurulu alarmlarım neler": {"cmd": "reminder.list", "args": {}},
        "yarın takvimimde ne var": {"cmd": "calendar.list", "args": {"day": "d+1"}},
        "zeynep'ten e-posta geldi mi": {"cmd": "mail.check", "args": {"query": "zeynep"}},
        "yeni e-postam var mı": {"cmd": "mail.check", "args": {}},
        "alışveriş listeme süt ekle": {"cmd": "note.write", "args": {"text": "süt", "tags": ["alışveriş"]}},
        "on iki artı yedi kaç eder": {"cmd": "calc.eval", "args": {"expr": "12+7"}},
        "everest dağı kaç metre": {"cmd": "ask", "args": {"question": "everest dağı kaç metre"}},
        "salondaki ışıkları kapat": {"cmd": "ask", "args": {"question": "salondaki ışıkları kapat"}},
        "yarın yağmur yağacak mı": {"cmd": "ask", "args": {"question": "yarın yağmur yağacak mı"}},
        "yarın sabah 9'da faturayı ödemeyi hatırlat": {"cmd": "reminder.set", "args": {"at": "d+1@09:00", "text": "faturayı ödemek"}},
        "alarmlarımı göster": {"cmd": "reminder.list", "args": {}},
        "alarmı iptal et": {"cmd": "ask", "args": {"question": "alarmı iptal et"}},
    }
    for src, w in want.items():
        assert src in rows, src
        assert json.loads(rows[src]["tgt"]) == w, (src, rows[src]["tgt"])
    # uydurma argüman yok: atlananlar hiçbir dosyada yok
    for src in ("beni saat beşte uyandır", "sabah yedide alarm kur", "ayşe ile yarın öğlen toplantı ekle", "tarkan çal",
                "yarınki alarmı sil", "bana bir fıkra anlat", "şarkı çal"):
        assert src not in rows, src
    sk = stats["massive"]["skipped"]
    assert sk["quality"] == 2 and sk["approximate_intent"] == 1 and sk["no_text"] == 2 and sk["residual_words"] == 1
    assert stats["xsid"]["skipped"]["approximate_intent"] == 1
    # olumsuz/sohbet ayrımı
    assert rows["salondaki ışıkları kapat"]["kind"] == "neg" and rows["everest dağı kaç metre"]["kind"] == "chat"


def test_every_row_is_valid_command(prepared):
    out, _ = prepared
    for f in (M.TRAIN_FILE, *M.EVAL_FILES.values()):
        for r in M._read(out / f):
            obj = json.loads(r["tgt"])
            validate_request({"v": 1, "id": "t", "cmd": obj["cmd"], "args": obj["args"]})
            assert r["license"] in (M.MASSIVE_LICENSE, M.XSID_LICENSE)


def test_train_eval_disjoint(prepared):
    out, stats = prepared
    train = M._read(out / M.TRAIN_FILE)
    assert train and all(r["partition"] == "train" and r["training_allowed"] is True for r in train)
    keys = {M.norm_key(r["src"]) for f in M.EVAL_FILES.values() for r in M._read(out / f)}
    assert not {M.norm_key(r["src"]) for r in train} & keys
    # dev ile birebir, büyük harf/noktalama farkıyla ve xSID ile çakışan train satırları düştü
    assert stats["massive"]["train_dropped"]["leak_eval"] == 3
    assert all(r["training_allowed"] is False for f in M.EVAL_FILES.values() for r in M._read(out / f))
    assert not [r for r in M._read(out / M.EVAL_FILES["xsid_tr"]) if r["license"] != M.XSID_LICENSE]
    # eğitim yükleyicisi ikinci kez denetler
    assert len(M.load_train(out)) == len(train)


def test_load_train_fails_loudly_on_leak_or_missing(prepared, tmp_path):
    out, _ = prepared
    rows = M._read(out / M.TRAIN_FILE)
    ev = M._read(out / M.EVAL_FILES["massive_tr"])
    M._write(out / M.TRAIN_FILE, rows + [{**ev[0], "training_allowed": True, "kind": "neg"}])
    with pytest.raises(M.DataError):
        M.load_train(out)
    (out / M.EVAL_FILES["xsid_tr"]).unlink()
    M._write(out / M.TRAIN_FILE, rows)
    with pytest.raises(M.DataError):
        M.load_train(out)
    with pytest.raises(M.DataError):
        M.load_train(tmp_path / "yok")


def test_sha_mismatch_fails_loudly(tmp_path):
    tar, sha = _tar(tmp_path)
    xd, xsha = _xsid(tmp_path)
    with pytest.raises(M.DataError):
        M.prepare(tmp_path / "o1", tar, xd, massive_sha="0" * 64, xsid_sha=xsha)
    with pytest.raises(M.DataError):
        M.prepare(tmp_path / "o2", tar, xd, massive_sha=sha, xsid_sha={k: "0" * 64 for k in xsha})
    # indirme kapalıyken önbellekte dosya yoksa: hata (sessiz geri dönüş yok)
    with pytest.raises(M.DataError):
        M.prepare(tmp_path / "o3")


def test_held_out_literals_match_tur18_list():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t18", Path(__file__).parent / "test_datagen_tur18.py")
    t18 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t18)
    assert M.HELD_OUT_LITERALS == t18.REMOVED_LITERALS


def test_stream_mix_share_and_unchanged_without_ext(prepared):
    pytest.importorskip("torch")
    from orion_core import datagen
    from orion_core.train import Stream
    out, _ = prepared
    ext = M.load_train(out)
    # dış veri yoksa akış eskisiyle aynı: rng yalnızca datagen tarafından tüketilir
    s = Stream(None, 192, 7, [], 0.2, None)
    rng = random.Random(7)
    for _ in range(50):
        a = s.example()
        r = rng.random()
        b = (datagen.t2c_example(rng, "train") if r < 0.8 else datagen.plan_example(rng, "train") if r < 0.92
             else datagen.answer_example(rng, "train"))
        assert a == b
    s = Stream(None, 192, 7, [], 0.2, None, ext_rows=ext, ext_frac=M.EXT_FRAC)
    srcs = {r["src"] for r in ext}
    share = sum(s.example()["src"] in srcs for _ in range(4000)) / 4000
    assert abs(share - M.EXT_FRAC) < 0.03


def test_score_ext_counts_wrong_actions():
    rows = [M.make_row("a", "ask", {"question": "ışık"}, kind="neg"), M.make_row("b", "ask", {"question": "ses"}, kind="neg"),
            M.make_row("c", "reminder.list", {}, kind="action")]
    from orion_core.evaluate import normalize_cmd
    outs = ['{"args":{"question":"ışık"},"cmd":"ask"}', '{"args":{},"cmd":"help"}', "bozuk"]
    sc = M.score_ext(rows, outs, normalize_cmd)
    assert sc["exact"] == pytest.approx(1 / 3) and sc["wrong_actions"] == pytest.approx(1 / 3)
    assert sc["valid"] == pytest.approx(2 / 3) and sc["per_kind"] == {"action": 0.0, "neg": 0.5}


def test_attribution_file_and_license_notes(tmp_path):
    text = (SHARED / "core-model" / "DIS-VERI.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in text and "CC BY-SA 4.0" in text and M.MASSIVE_SHA256 in text
    from orion_core import pipeline
    p = tmp_path / "LN.md"
    pipeline.write_license_notes(p)
    ln = p.read_text(encoding="utf-8")
    assert "MASSIVE" in ln and "xSID" in ln
    # Eşitleme betiği yalnızca işlem deposunda var; halka açık depoda bu kontrol atlanır.
    sync_path = SHARED.parent / "tools" / "gelisim-sync.sh"
    if sync_path.exists():
        assert "DIS-VERI.md" in sync_path.read_text(encoding="utf-8")


REAL = Path(os.environ.get("ORION_EXT_DATA", "")) if os.environ.get("ORION_EXT_DATA") else None


@pytest.mark.skipif(not (REAL and (REAL / M.TRAIN_FILE).exists()), reason="gerçek MASSIVE/xSID hazırlanmadı (ağsız test)")
def test_real_data_disjoint_and_valid():
    train = M.load_train(REAL)  # sızıntı varsa DataError
    assert len(train) > 1000
    for f in (M.TRAIN_FILE, *M.EVAL_FILES.values()):
        for r in M._read(REAL / f):
            obj = json.loads(r["tgt"])
            validate_request({"v": 1, "id": "t", "cmd": obj["cmd"], "args": obj["args"]})
    for r in train:
        assert not any(lit in r["src"] for lit in M.HELD_OUT_LITERALS)
