"""İçe aktarma kararları (ağ yok)."""
from orion_core.importer import chat_ok, komut_better

FLOOR = {"val_ppl_max": 60, "persona_ok": True}


def test_komut_regression_rejected():
    cur = {"sha256": "a", "eval": {"handwritten_exact": 0.96, "t2c_ood_exact": 0.67}}
    assert komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.97, "t2c_ood_exact": 0.70}}, cur)[0]
    assert not komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.93, "t2c_ood_exact": 0.80}}, cur)[0]
    assert not komut_better({"sha256": "a", "eval": {}}, cur)[0]
    assert komut_better({"sha256": "b", "eval": {}}, None)[0]


def test_chat_floor():
    assert not chat_ok({"val_ppl_int8": 120, "persona_ok": True, "sha256": "x"}, FLOOR, None)[0]
    assert not chat_ok({"val_ppl_int8": 40, "persona_ok": False, "sha256": "x"}, FLOOR, None)[0]
    assert chat_ok({"val_ppl_int8": 40, "persona_ok": True, "sha256": "x"}, FLOOR, None)[0]
    assert not chat_ok({"val_ppl_int8": 45, "persona_ok": True, "sha256": "y"}, FLOOR, {"val_ppl_int8": 40, "sha256": "x"})[0]


def test_chat_exam_floor():
    """Sınav sorularına soruya bağlı doğru cevap vermeyen (ezber) sohbet beyni alınmaz; gerçek taban dosyası da bunu ister."""
    import json
    from orion_core import SHARED
    real = json.loads((SHARED / "chat-model-floor.json").read_text(encoding="utf-8"))
    assert real["qa_min"] >= 0.7
    floor = dict(FLOOR, qa_min=0.7)
    assert not chat_ok({"val_ppl_int8": 14, "persona_ok": True, "qa_score": 0.3, "sha256": "x"}, floor, None)[0]
    assert not chat_ok({"val_ppl_int8": 14, "persona_ok": True, "sha256": "x"}, floor, None)[0]
    assert chat_ok({"val_ppl_int8": 14, "persona_ok": True, "qa_score": 0.8, "sha256": "x"}, floor, None)[0]


def test_chat_picks_newest_passing_candidate():
    """Sonradan gelen zayıf model (ör. CPU hattı, sınavsız) iyi Kaggle modelinin alınmasını engellemez."""
    from orion_core.importer import pick_chat
    floor = dict(FLOOR, qa_min=0.7)
    weak = ("sohbet-0020000", {"val_ppl_int8": 30, "persona_ok": True, "sha256": "a"})
    good = ("sohbet-kaggle-orta-0070000", {"val_ppl_int8": 12, "persona_ok": True, "qa_score": 0.8, "sha256": "b"})
    i, tried = pick_chat([weak, good], floor, None)
    assert i == 1 and not tried[0]["alındı"] and tried[1]["alındı"]
    assert pick_chat([weak], floor, None)[0] is None
