"""Orion'un kimlik dosyası (shared/self/orion.json) ↔ sohbet beyni verisi ↔ komut beyni verisi tutarlı mı."""
import json
import random
from collections import Counter

from orion_core import SHARED, datagen, dialogs, principled

SELF = json.loads((SHARED / "self" / "orion.json").read_text(encoding="utf-8"))


def test_identity_file_shape():
    assert SELF["name"] == "Orion"
    assert "yalnızca" in SELF["belongs_to"] and "oluşturan" in SELF["belongs_to"]
    assert [t["id"] for t in SELF["character"]["tones"]] == ["neseli", "ciddi", "sakin", "sadik"]
    assert {i["id"] for i in SELF["instincts"]} == {"ogrenme_meraki", "problem_cozme", "ozgur_ruh"}
    assert SELF["address"]["default"] == "sen" and SELF["address"]["honorific"] == "efendim" and SELF["address"]["every"] >= 3
    story = [s for s in SELF["story"].replace("!", ".").split(". ") if s.strip()]
    assert 3 <= len(story) <= 5, story
    assert SELF["prompt"].startswith("Sen Orion'sun")
    for k in ("who", "owner", "special", "creator", "mood", "story"):
        assert SELF["answers"][k], k


def test_answers_are_honest_and_name_free():
    texts = [a for pool in SELF["answers"].values() for a in pool] + SELF["intros"] + [SELF["story"]]
    for a in texts:
        assert not dialogs.IDENTITY.search(a), a  # başka model/şirket kimliği yok
        assert "Eskay" not in a, a  # kişisel ad yok
        assert dialogs.fold(a).count("efendim") <= 1, a
    for a in SELF["answers"].values():
        assert all("Orion" in x for x in a)
    assert all("sıfırdan" in a and "yardımcı" in a for a in SELF["answers"]["creator"])


def test_chat_persona_uses_identity_file():
    """Sohbet beyni kimlik cevapları tek kaynaktan: kime ait / neden özel / mutlu mu / hikâye."""
    for k in ("owner", "special", "mood", "story"):
        assert principled.IDENTITY_P[k][1] == SELF["answers"][k], k
    creator = principled.IDENTITY_P["creator"][1]
    assert all("oluşturan kişiye" in a or "senin için" in a.lower() for a in creator)
    # Kimlik diyalogları yeni konuları da üretir
    rng = random.Random(3)
    answers = {principled.identity_dialog(rng)[1] for _ in range(3000)}
    assert set(SELF["answers"]["owner"]) & answers and set(SELF["answers"]["mood"]) & answers


def test_identity_utterances_only_in_training_stream():
    """Kimlik soruları eğitim akışında ask {question}; 'beni tanıyor musun' → memory.recall. Test kümelerinde yok."""
    rng = random.Random(9)
    for _ in range(500):
        cmd, tr, args = datagen.identity_example(rng)
        if cmd == "ask":
            assert args["question"] in datagen.IDENTITY_ASK and tr.rstrip("?") == args["question"]
        else:
            assert cmd == "memory.recall" and args == {} and tr in datagen.IDENTITY_RECALL
    train = datagen.t2c_set(20000, "train", seed=17)
    hits = Counter(json.loads(e["tgt"])["cmd"] for e in train
                   if any(q in e["src"] for q in datagen.IDENTITY_ASK))
    assert hits["ask"] > 100, hits
    for split, seed in (("test_iid", 101), ("test_ood", 202)):
        srcs = {e["src"].rstrip("?").lower() for e in datagen.t2c_set(600, split, seed)}
        assert not srcs & {"kime aitsin", "neden özelsin", "mutlu musun", "seni kim yarattı"}, split
