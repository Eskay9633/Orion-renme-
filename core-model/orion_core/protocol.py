"""Orion makine dili v1: ayrıştırıcı, şema doğrulayıcı, kanonik biçim.

Kural seti shared/protocol/v1/commands.json dosyasından okunur. Kotlin tarafı
(orion-core modülü) aynı dosyayı ve aynı test vektörlerini kullanır.
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import SHARED

SCHEMA_PATH = SHARED / "protocol" / "v1" / "commands.json"

ID_RE = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")
REPO_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9\-]{0,38})/[A-Za-z0-9._\-]{1,100}$")
URI_RE = re.compile(r"^(content|file|https)://\S{1,2000}$")
HHMM = r"([0-9]{2}):([0-9]{2})"
WHEN_ABS = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})T" + HHMM + "$")
WHEN_OFFSET = re.compile(r"^\+([0-9]{1,4})([mhd])$")
WHEN_DAY = re.compile(r"^d\+([0-9]{1,3})@" + HHMM + "$")
WHEN_DOW = re.compile(r"^(mon|tue|wed|thu|fri|sat|sun)@" + HHMM + "$")
WHEN_MD = re.compile(r"^([0-9]{2})-([0-9]{2})@" + HHMM + "$")
DATE_ABS = re.compile(r"^([0-9]{4})-([0-9]{2})-([0-9]{2})$")
DATE_REL = re.compile(r"^d\+([0-9]{1,3})$")

REQUEST_KEYS = {"v", "id", "cmd", "args", "ctx"}
RESPONSE_KEYS = {"v", "id", "status", "result", "err_code", "next"}
MAX_NEXT = 8


class ProtocolError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Schema:
    version: int
    commands: dict[str, Any]
    agents: tuple[str, ...]
    statuses: tuple[str, ...]
    error_codes: dict[str, str]


@lru_cache(maxsize=4)
def load_schema(path: str | None = None) -> Schema:
    raw = json.loads(Path(path or SCHEMA_PATH).read_text(encoding="utf-8"))
    return Schema(
        version=raw["v"],
        commands=raw["commands"],
        agents=tuple(raw["agents"]),
        statuses=tuple(raw["statuses"]),
        error_codes=raw["error_codes"],
    )


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_hm(h: str, m: str) -> None:
    if int(h) > 23 or int(m) > 59:
        raise ProtocolError("E_ARG_RANGE", "saat/dakika sınır dışı")


def _check_date(y: int, mo: int, d: int) -> None:
    if not (1 <= mo <= 12):
        raise ProtocolError("E_ARG_RANGE", "ay sınır dışı")
    if not (1 <= d <= calendar.monthrange(y, mo)[1]):
        raise ProtocolError("E_ARG_RANGE", "gün sınır dışı")


def check_when(s: str) -> None:
    if m := WHEN_ABS.match(s):
        _check_date(int(m[1]), int(m[2]), int(m[3]))
        _check_hm(m[4], m[5])
    elif m := WHEN_OFFSET.match(s):
        if int(m[1]) < 1:
            raise ProtocolError("E_ARG_RANGE", "sıfır ofset")
    elif m := WHEN_DAY.match(s):
        _check_hm(m[2], m[3])
    elif m := WHEN_DOW.match(s):
        _check_hm(m[2], m[3])
    elif m := WHEN_MD.match(s):
        _check_date(2024, int(m[1]), int(m[2]))  # artık yıl: 02-29 geçerli
        _check_hm(m[3], m[4])
    else:
        raise ProtocolError("E_ARG_TYPE", f"zaman biçimi geçersiz: {s!r}")


def check_date(s: str) -> None:
    if m := DATE_ABS.match(s):
        _check_date(int(m[1]), int(m[2]), int(m[3]))
    elif not DATE_REL.match(s):
        raise ProtocolError("E_ARG_TYPE", f"tarih biçimi geçersiz: {s!r}")


def _validate_arg(name: str, spec: dict, val: Any) -> None:
    t = spec["type"]
    if t in ("string", "enum", "when", "date", "repo", "uri"):
        if not isinstance(val, str):
            raise ProtocolError("E_ARG_TYPE", f"{name} metin olmalı")
    if t == "string":
        if len(val) < spec.get("minLen", 0) or len(val) > spec.get("maxLen", 10_000):
            raise ProtocolError("E_ARG_RANGE", f"{name} uzunluğu sınır dışı")
    elif t == "enum":
        if val not in spec["values"]:
            raise ProtocolError("E_ARG_RANGE", f"{name} izinli değerlerden biri değil")
    elif t == "int":
        if not _is_int(val):
            raise ProtocolError("E_ARG_TYPE", f"{name} tam sayı olmalı")
        if val < spec.get("min", -(2**31)) or val > spec.get("max", 2**31 - 1):
            raise ProtocolError("E_ARG_RANGE", f"{name} sınır dışı")
    elif t == "bool":
        if not isinstance(val, bool):
            raise ProtocolError("E_ARG_TYPE", f"{name} true/false olmalı")
    elif t == "when":
        check_when(val)
    elif t == "date":
        check_date(val)
    elif t == "repo":
        if not REPO_RE.match(val):
            raise ProtocolError("E_ARG_TYPE", f"{name} sahip/isim biçiminde olmalı")
    elif t == "uri":
        if not URI_RE.match(val):
            raise ProtocolError("E_ARG_TYPE", f"{name} geçerli bir adres değil")
    elif t == "strlist":
        if not isinstance(val, list):
            raise ProtocolError("E_ARG_TYPE", f"{name} liste olmalı")
        if len(val) > spec.get("maxItems", 100):
            raise ProtocolError("E_ARG_RANGE", f"{name} çok uzun")
        for item in val:
            if not isinstance(item, str):
                raise ProtocolError("E_ARG_TYPE", f"{name} öğeleri metin olmalı")
            if not (1 <= len(item) <= spec.get("itemMaxLen", 200)):
                raise ProtocolError("E_ARG_RANGE", f"{name} öğe uzunluğu sınır dışı")
    else:  # pragma: no cover - şema hatası
        raise ProtocolError("E_INTERNAL", f"bilinmeyen tür {t}")


def validate_request(obj: Any, schema: Schema | None = None) -> dict:
    """Komutu doğrular ve varsayılanları doldurulmuş kopyasını döndürür."""
    schema = schema or load_schema()
    if not isinstance(obj, dict):
        raise ProtocolError("E_SCHEMA", "kök nesne olmalı")
    extra = set(obj) - REQUEST_KEYS
    if extra:
        raise ProtocolError("E_SCHEMA", f"tanımsız alan: {sorted(extra)}")
    for key in ("v", "id", "cmd", "args"):
        if key not in obj:
            raise ProtocolError("E_SCHEMA", f"{key} eksik")
    if not _is_int(obj["v"]):
        raise ProtocolError("E_SCHEMA", "v tam sayı olmalı")
    if obj["v"] != schema.version:
        raise ProtocolError("E_VERSION", f"v={obj['v']}")
    if not isinstance(obj["id"], str) or not ID_RE.match(obj["id"]):
        raise ProtocolError("E_SCHEMA", "id geçersiz")
    if not isinstance(obj["cmd"], str):
        raise ProtocolError("E_SCHEMA", "cmd metin olmalı")
    if not isinstance(obj["args"], dict):
        raise ProtocolError("E_SCHEMA", "args nesne olmalı")
    ctx = obj.get("ctx", {})
    if not isinstance(ctx, dict):
        raise ProtocolError("E_SCHEMA", "ctx nesne olmalı")
    if "agent" in ctx and ctx["agent"] not in schema.agents:
        raise ProtocolError("E_SCHEMA", "ctx.agent geçersiz")
    spec = schema.commands.get(obj["cmd"])
    if spec is None:
        raise ProtocolError("E_UNKNOWN_CMD", obj["cmd"])
    arg_specs: dict = spec["args"]
    unknown = set(obj["args"]) - set(arg_specs)
    if unknown:
        raise ProtocolError("E_ARG_UNKNOWN", f"{sorted(unknown)}")
    args: dict = {}
    for name, aspec in arg_specs.items():
        if name in obj["args"]:
            _validate_arg(name, aspec, obj["args"][name])
            args[name] = obj["args"][name]
        elif aspec.get("required"):
            raise ProtocolError("E_ARG_MISSING", name)
        elif "default" in aspec:
            args[name] = aspec["default"]
    return {"v": obj["v"], "id": obj["id"], "cmd": obj["cmd"], "args": args, "ctx": dict(ctx)}


def validate_response(obj: Any, schema: Schema | None = None) -> dict:
    schema = schema or load_schema()
    if not isinstance(obj, dict):
        raise ProtocolError("E_SCHEMA", "kök nesne olmalı")
    if set(obj) != RESPONSE_KEYS:
        raise ProtocolError("E_SCHEMA", "alanlar: v,id,status,result,err_code,next")
    if not _is_int(obj["v"]):
        raise ProtocolError("E_SCHEMA", "v tam sayı olmalı")
    if obj["v"] != schema.version:
        raise ProtocolError("E_VERSION", f"v={obj['v']}")
    if not isinstance(obj["id"], str) or not ID_RE.match(obj["id"]):
        raise ProtocolError("E_SCHEMA", "id geçersiz")
    if obj["status"] not in schema.statuses:
        raise ProtocolError("E_SCHEMA", "status geçersiz")
    if obj["result"] is not None and not isinstance(obj["result"], dict):
        raise ProtocolError("E_SCHEMA", "result nesne ya da null olmalı")
    err = obj["err_code"]
    if err is not None and err not in schema.error_codes:
        raise ProtocolError("E_SCHEMA", "err_code bilinmiyor")
    if obj["status"] == "error" and err is None:
        raise ProtocolError("E_SCHEMA", "error durumunda err_code zorunlu")
    if obj["status"] != "error" and err is not None:
        raise ProtocolError("E_SCHEMA", "err_code yalnızca error durumunda")
    if not isinstance(obj["next"], list) or len(obj["next"]) > MAX_NEXT:
        raise ProtocolError("E_SCHEMA", "next en çok 8 komutluk liste olmalı")
    nxt = [validate_request(n, schema) for n in obj["next"]]
    out = dict(obj)
    out["next"] = nxt
    return out


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError) as e:
        raise ProtocolError("E_PARSE", str(e)) from None


def parse_request(text: str) -> dict:
    return validate_request(parse(text))


def parse_response(text: str) -> dict:
    return validate_response(parse(text))


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def make_response(req_id: str, status: str, result: dict | None = None,
                  err_code: str | None = None, nxt: list | None = None) -> dict:
    return {"v": 1, "id": req_id, "status": status, "result": result,
            "err_code": err_code, "next": nxt or []}
