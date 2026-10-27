import json

from orion_core import datagen
from orion_core.protocol import validate_request


def test_all_generated_commands_validate():
    for split in ("train", "test_iid", "test_ood"):
        for ex in datagen.generate(3000, split, seed=1):
            if ex["task"] == "t2c":
                obj = json.loads(ex["tgt"])
                validate_request({"v": 1, "id": "x", "cmd": obj["cmd"], "args": obj["args"]})
            assert ex["src"] and ex["tgt"]


def test_every_command_covered():
    seen = {json.loads(e["tgt"])["cmd"] for e in datagen.t2c_set(5000, "train", seed=2)}
    assert seen == set(datagen.TEMPLATES)


def test_deterministic():
    assert datagen.generate(200, "train", seed=5) == datagen.generate(200, "train", seed=5)


def test_iid_values_unseen_in_train():
    train_texts = set(datagen._split(datagen.REMINDER_TEXTS, "train"))
    iid_texts = set(datagen._split(datagen.REMINDER_TEXTS, "test_iid"))
    assert train_texts.isdisjoint(iid_texts)


def test_ood_templates_differ():
    tr = {e["src"] for e in datagen.t2c_set(4000, "train", seed=3)}
    ood = [e for e in datagen.t2c_set(500, "test_ood", seed=4)]
    assert sum(e["src"] in tr for e in ood) / len(ood) < 0.3


def test_locative_suffix():
    assert [datagen.loc_suffix(n) for n in (1, 3, 6, 9, 10, 14, 20, 30, 40, 45, 50, 16)] == \
        ["de", "te", "da", "da", "da", "te", "de", "da", "ta", "te", "de", "da"]


def test_plan_targets():
    p = json.loads(datagen.plan_for({"cmd": "research.start", "args": {"topic": "x", "depth": "deep"}}))
    assert p["agent"] == "sophia" and "provider.judge" in p["steps"]


def test_training_stream_never_crashes_on_protocol_limits():
    """Eğitim akışı rastgele metin üretir; protokol sınırını aşan örnek atlanır, eğitim çökmez (komut-beyni #7 hatası)."""
    import random
    from orion_core import datagen
    rng = random.Random(12345)
    for _ in range(30000):
        datagen.t2c_example(rng, "train")
