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
