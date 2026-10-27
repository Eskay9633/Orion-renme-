"""Veri hattı: kaynak etiketleme, gizli veri temizleme, tekilleştirme, eğitim filtresi,
LICENSE-NOTES.md ve veri/kaynak-kayit.md üretimi.

Kayıt biçimi (jsonl, her satır bir örnek):
  task, src, tgt              görev ve çift
  provider, model, date       kaynak sağlayıcı / model / tarih
  license                     lisans notu
  training_allowed            yalnızca true olanlar eğitime girer
  origin                      grammar | api | manual_import | local | user
  user_rating                 -1 / 0 / +1 (kullanıcı puanı)
Kullanıcının onay/ret verisi (DPO için) ayrı dosyada tutulur: prefs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from . import REPO_ROOT, SHARED
from .protocol import ProtocolError, canonical, validate_request
from .scrub import contains_hard_secret, scrub

REQUIRED = ("task", "src", "tgt", "provider", "model", "date", "license", "training_allowed", "origin")
PROVIDERS = SHARED / "providers" / "providers.json"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def provider_defaults() -> dict[str, dict]:
    return {p["id"]: p for p in json.loads(PROVIDERS.read_text(encoding="utf-8"))["providers"]}


def clean_record(r: dict, providers: dict[str, dict]) -> dict | None:
    """Tek kaydı temizler; geçersizse None. Kural: manual_import ve kapalı sağlayıcılar eğitime giremez
    (kullanıcı bir sağlayıcı için açıkça izin vermedikçe — o izin kayda training_allowed=true olarak yazılır)."""
    if any(k not in r for k in REQUIRED):
        return None
    r = dict(r)
    if r["origin"] == "manual_import":
        r["training_allowed"] = False
    p = providers.get(r["provider"])
    if p is not None and p.get("api") == "manual":
        r["training_allowed"] = False
    if r["training_allowed"] is not True:
        r["training_allowed"] = False
    # Sert gizli (anahtar/parola/kart/IBAN/TCKN) içeren örnek hiç tutulmaz; diğer kişisel veri maskelenir.
    if contains_hard_secret(r["src"]) or contains_hard_secret(r["tgt"]):
        return None
    r["src"] = scrub(r["src"]).strip()
    r["tgt"] = r["tgt"].strip()
    if not r["src"] or not r["tgt"]:
        return None
    if r["task"] == "t2c":
        try:
            obj = json.loads(r["tgt"])
            validate_request({"v": 1, "id": "p", "cmd": obj["cmd"], "args": obj["args"]})
            r["tgt"] = canonical({"cmd": obj["cmd"], "args": obj["args"]})
        except (json.JSONDecodeError, ProtocolError, KeyError, TypeError):
            return None
    return r


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        key = hashlib.sha1(f"{r['task']}\x00{' '.join(r['src'].lower().split())}\x00{r['tgt']}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def training_rows(rows: list[dict]) -> list[dict]:
    """Eğitime yalnızca training_allowed=true giren örnekler (filtre testle korunur)."""
    return [r for r in rows if r.get("training_allowed") is True]


def process(raw_paths: list[Path], out_dir: Path) -> dict:
    providers = provider_defaults()
    raw = [r for p in raw_paths for r in load_jsonl(p)]
    cleaned = [c for r in raw if (c := clean_record(r, providers)) is not None]
    uniq = dedupe(cleaned)
    train = training_rows(uniq)
    write_jsonl(out_dir / "all.jsonl", uniq)
    write_jsonl(out_dir / "train.jsonl", train)
    return {"raw": len(raw), "clean": len(cleaned), "unique": len(uniq), "trainable": len(train),
            "by_provider": dict(Counter(r["provider"] for r in uniq)),
            "trainable_by_provider": dict(Counter(r["provider"] for r in train))}


# --------------------------------------------------------------------------- belgeler

def write_license_notes(path: Path, stats: dict | None = None) -> None:
    providers = json.loads(PROVIDERS.read_text(encoding="utf-8"))
    lines = [
        "# LICENSE-NOTES — Sağlayıcı ve model lisans notları",
        "",
        "> Bu dosya `python -m orion_core.pipeline --docs` ile **otomatik** üretilir; elle düzenleme.",
        f"> Kaynak: `shared/providers/providers.json` (kontrol tarihi: {providers['checked_on']}).",
        "> **Doğrulanmadı** yazan her bilgi, kullanmadan önce sağlayıcının güncel sayfasından kontrol edilmelidir.",
        "",
        "Kural: eğitime yalnızca `training_allowed=true` örnekler girer. Kapalı modeller ve elle içe",
        "aktarılanlar varsayılan olarak `false`'tur. Kullanıcı bunu sağlayıcı başına, uyarıyı onaylayarak",
        "değiştirebilir; bu durumda sorumluluk kullanıcıdadır.",
        "",
        "| Sağlayıcı | Tür | training_allowed | Otomatik erişim | Veriyi eğitimde kullanır mı | Gizli veri | Doğrulandı | Lisans notu |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tr_kind = {"open_weight": "açık ağırlık", "closed": "kapalı", "local": "yerel", "manual": "elle içe aktar"}
    tr_yn = {"yes": "evet", "no": "hayır", "unknown": "bilinmiyor"}
    for p in providers["providers"]:
        lines.append(
            f"| {p['name']} | {tr_kind[p['kind']]} | {'evet' if p['training_allowed'] else 'hayır'} | "
            f"{'evet' if p['auto_access'] else 'HAYIR'} | {tr_yn[p['provider_trains_on_data']]} | "
            f"{'evet' if p['secret_data_allowed'] else 'hayır'} | {'evet' if p['verified'] else '**doğrulanmadı**'} | "
            f"{p['license_note']} |")
    lines += ["", "## Notlar", ""]
    for p in providers["providers"]:
        lines.append(f"- **{p['name']}**: {p['notes']}")
    lines += [
        "", "## Orion'un kendi bileşenleri", "",
        "- **Çekirdek model ve tokenizer**: sıfırdan, yalnızca programla üretilmiş veriyle (+ izinli damıtma verisiyle) eğitilir. Ağırlıklar bu deponun lisansına tabidir.",
        "- **Programla üretilen veri** (`orion_core/datagen.py`): kota ve lisans sorunu yok.",
    ]
    if stats:
        lines += ["", "## Son veri hattı çalışması", "", "```", json.dumps(stats, ensure_ascii=False, indent=1), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_log(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = path.read_text(encoding="utf-8") if path.exists() else (
        "# Veri kaynak kaydı\n\nHer veri hattı çalışmasında otomatik eklenir: hangi sağlayıcıdan kaç örnek "
        "geldi, kaçı eğitime girdi.\n\n| Tarih | Ham | Temiz | Tekil | Eğitime giren | Sağlayıcı dağılımı |\n|---|---|---|---|---|---|\n")
    row = (f"| {date.today().isoformat()} | {stats['raw']} | {stats['clean']} | {stats['unique']} | "
           f"{stats['trainable']} | {', '.join(f'{k}:{v}' for k, v in sorted(stats['by_provider'].items())) or '-'} |\n")
    path.write_text(prev + row, encoding="utf-8")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", default=[], help="ham jsonl dosyaları")
    ap.add_argument("--out", default=str(REPO_ROOT / "core-model" / "work" / "data"))
    ap.add_argument("--docs", action="store_true", help="LICENSE-NOTES.md ve veri/kaynak-kayit.md güncelle")
    a = ap.parse_args(argv)
    stats = process([Path(p) for p in a.raw], Path(a.out)) if a.raw else None
    if stats:
        print(json.dumps(stats, ensure_ascii=False))
    if a.docs:
        write_license_notes(REPO_ROOT / "LICENSE-NOTES.md", stats)
        if stats:
            write_source_log(REPO_ROOT / "veri" / "kaynak-kayit.md", stats)


if __name__ == "__main__":
    main()
