import json

import pytest

from orion_core import SHARED
from orion_core.protocol import (ProtocolError, canonical, load_schema, parse_request,
                                 parse_response)

VECTORS = json.loads((SHARED / "protocol/v1/test-vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("vec", VECTORS["requests"], ids=lambda v: v["name"])
def test_request_vectors(vec):
    if vec["ok"]:
        req = parse_request(vec["input"])
        if "canonical" in vec:
            assert canonical(req) == vec["canonical"]
    else:
        with pytest.raises(ProtocolError) as e:
            parse_request(vec["input"])
        assert e.value.code == vec["err_code"]


@pytest.mark.parametrize("vec", VECTORS["responses"], ids=lambda v: v["name"])
def test_response_vectors(vec):
    if vec["ok"]:
        parse_response(vec["input"])
    else:
        with pytest.raises(ProtocolError) as e:
            parse_response(vec["input"])
        assert e.value.code == vec["err_code"]


def test_every_command_has_agent_and_doc():
    s = load_schema()
    for name, spec in s.commands.items():
        assert spec["agent"] in s.agents, name
        assert spec["doc"], name
        for aname, a in spec["args"].items():
            assert a["type"] in {"string", "int", "bool", "enum", "when", "date", "repo", "uri", "strlist"}
            if "default" in a:
                assert not a.get("required"), f"{name}.{aname} hem zorunlu hem varsayılanlı"


def test_defaults_are_valid():
    from orion_core.protocol import _validate_arg
    s = load_schema()
    for name, spec in s.commands.items():
        # Varsayılan değerler kendi doğrulayıcısından geçmeli
        for aname, a in spec["args"].items():
            if "default" in a:
                _validate_arg(aname, a, a["default"])
