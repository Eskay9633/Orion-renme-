import json

import pytest

from orion_core import SHARED
from orion_core.scrub import contains_secret, scrub

CASES = json.loads((SHARED / "scrub/test-vectors.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["in"][:30])
def test_scrub_vectors(case):
    assert scrub(case["in"]) == case["out"]
    assert contains_secret(case["in"]) == case["secret"]
