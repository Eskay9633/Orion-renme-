"""'Emin değilsem sorayım': commands.json effect alanı ve onay eşiği (confirm_threshold)."""
from orion_core.protocol import load_schema, validate_request

EFFECTS = {"read", "write-timed", "write-chain", "write-easy", "proposal"}


def test_every_command_has_effect():
    s = load_schema()
    for name, spec in s.commands.items():
        assert spec.get("effect") in EFFECTS, name
    assert s.commands["reminder.set"]["effect"] == "write-timed"
    assert s.commands["skill.run"]["effect"] == "write-chain"
    assert s.commands["note.write"]["effect"] == "write-easy"
    assert s.commands["mail.send"]["effect"] == "proposal"
    assert s.commands["ask"]["effect"] == "read"


def test_effect_field_does_not_change_validation():
    r = validate_request({"v": 1, "id": "t1", "cmd": "reminder.set", "args": {"at": "d+1@10:00", "text": "toplantı"}})
    assert r["args"]["repeat"] == "none"


def test_confirm_band_lowest_threshold_with_enough_accuracy():
    from orion_core.evaluate import confirm_band
    # 0.90–0.97 arası çoğu doğru, 0.70–0.80 arası çoğu yanlış
    good = [(0.90 + i * 0.001, i % 5 != 0) for i in range(60)]   # %80 doğru
    bad = [(0.70 + i * 0.001, i % 5 == 0) for i in range(60)]    # %20 doğru
    sure = [(0.99, True)] * 100
    out = confirm_band({"test_iid": good + bad + sure}, abstain=0.98)
    assert 0.7 <= out["threshold"] < 0.98
    assert out["band_accuracy"] >= 0.6
    assert not out["default_used"]
    # Eşiğin hemen altına inince doğruluk 0.6'nın altına düşmeli (en düşük uygun değer seçildi)
    assert out["threshold"] > 0.70


def test_confirm_band_defaults_to_08():
    from orion_core.evaluate import confirm_band
    out = confirm_band({"x": [(0.5, False)] * 30 + [(0.99, True)] * 30}, abstain=0.98)
    assert out["threshold"] == 0.8 and out["default_used"]
    # Eşik abstain'i geçemez
    assert confirm_band({"x": []}, abstain=0.6)["threshold"] == 0.6
