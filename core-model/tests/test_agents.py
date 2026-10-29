"""Ajan kimlikleri tek kaynakla tutarlı mı: shared/self/ajanlar.json ↔ commands.json ↔ datagen."""
import json
from pathlib import Path

from orion_core import datagen

SHARED = Path(__file__).resolve().parents[2] / "shared"


def _load(p):
    return json.loads((SHARED / p).read_text(encoding="utf-8"))


def test_agent_ids_match_protocol_and_datagen():
    agents = _load("self/ajanlar.json")["agents"]
    ids = [a["id"] for a in agents]
    proto = _load("protocol/v1/commands.json")
    assert ids == proto["agents"]
    assert sorted(aid for _, aid, _ in datagen.AGENT_TR) == sorted(ids)
    assert sorted(n.lower() for n in datagen.AGENT_NAMES if n != "Orion") == sorted(ids)
    assert {a["name"].lower() for a in agents} == set(ids)


def test_agent_commands_and_principles_exist():
    agents = _load("self/ajanlar.json")["agents"]
    proto = _load("protocol/v1/commands.json")
    cmds = proto["commands"]
    cmd_ids = set(cmds) if isinstance(cmds, dict) else {c.get("cmd") or c.get("id") for c in cmds}
    principle_ids = {p["id"] for p in _load("self/ilkeler.json")["principles"]}
    by_id = {a["id"]: a for a in agents}
    for a in agents:
        assert set(a["commands"]) <= cmd_ids, a["id"]
        assert set(a["principles"]) <= principle_ids, a["id"]
        assert {"kendini_tani", "durust_ol"} <= set(a["principles"]), a["id"]
    items = cmds.items() if isinstance(cmds, dict) else [(c.get("cmd") or c.get("id"), c) for c in cmds]
    for cid, spec in items:
        assert cid in by_id[spec["agent"]]["commands"], cid
