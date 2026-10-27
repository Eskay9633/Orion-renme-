"""Damıtma verisi toplama (öğretmen modellerden, yalnızca izinli sağlayıcılar).

İki damıtma türü var:
  1) VERİ damıtması (bu dosya): training_allowed=true sağlayıcılara, bir komutun doğal Türkçe
     söyleyişlerini yazdırırız. Komutu BİZ üretiriz (doğrulanmış); öğretmen yalnızca Türkçe'yi
     çeşitlendirir. Serbest metin değerleri cümlede birebir geçmeyen satırlar atılır.
  2) LOGİT damıtması (train.py --teacher): büyük Orion öğretmeninin yumuşak olasılıkları
     küçük öğrenciye aktarılır. Tamamen çevrimdışı, şart/lisans sorunu yok.

Anahtarlar ortam değişkeninden okunur (GitHub Secrets): ORION_KEY_GROQ, ORION_KEY_OPENROUTER ...
Anahtar yoksa iş sessizce atlanır; depoya asla anahtar yazılmaz.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from . import datagen
from .pipeline import provider_defaults

PROMPT = (
    "Aşağıdaki makine dili komutunu, bir kullanıcının telefonuna yazacağı doğal Türkçe isteğe çevir. "
    "{k} farklı söyleyiş yaz, her biri ayrı satırda, numara ya da açıklama ekleme. "
    "Tırnak içindeki değerleri AYNEN kullan. Zaman ifadeleri: d+1@10:00 = yarın 10:00, +15m = 15 dakika sonra, "
    "fri@14:00 = cuma 14:00, 10-29@08:00 = 29 Ekim 08:00, d+0 = bugün.\nKomut: {cmd}"
)
FREE = {"text", "title", "topic", "query", "label", "place", "need", "repo"}


def eligible(providers: dict) -> list[dict]:
    out = []
    for p in providers.values():
        key = os.environ.get(f"ORION_KEY_{p['id'].upper()}")
        if key and p["training_allowed"] and p["auto_access"] and p["api"] == "openai" and p["kind"] != "manual":
            out.append({**p, "_key": key})
    return out


def ask(p: dict, content: str, timeout: int = 60) -> str:
    base = p["base_url"].replace("{account_id}", os.environ.get(f"ORION_ACCOUNT_{p['id'].upper()}", ""))
    body = json.dumps({"model": p["model"], "messages": [{"role": "user", "content": content}],
                       "max_tokens": 400, "temperature": 0.9}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {p['_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def keep_line(line: str, args: dict) -> bool:
    line = line.strip()
    if not (4 <= len(line) <= 300) or line.startswith(("{", "Komut", "-", "*")):
        return False
    return all(str(v).lower() in line.lower() for k, v in args.items() if k in FREE and isinstance(v, str))


def collect(n_prompts: int, per_prompt: int, out: Path, seed: int, max_per_provider: int) -> dict:
    providers = provider_defaults()
    teachers = eligible(providers)
    if not teachers:
        print("Öğretmen yok: training_allowed=true ve anahtarı tanımlı sağlayıcı bulunamadı; atlanıyor.")
        return {"teachers": 0, "rows": 0}
    rng = random.Random(seed)
    used = {t["id"]: 0 for t in teachers}
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(n_prompts):
        live = [t for t in teachers if used[t["id"]] < max_per_provider]
        if not live:
            break
        t = live[i % len(live)]
        ex = datagen.t2c_example(rng, "train", noise=False)
        cmd = json.loads(ex["tgt"])
        try:
            text = ask(t, PROMPT.format(k=per_prompt, cmd=ex["tgt"]))
        except urllib.error.HTTPError as e:
            print(f"{t['id']}: HTTP {e.code}; bu sağlayıcı bugün için bırakılıyor")
            used[t["id"]] = max_per_provider
            continue
        except Exception as e:  # ağ hatası
            print(f"{t['id']}: {e}")
            continue
        used[t["id"]] += 1
        for line in text.splitlines():
            if keep_line(line, cmd["args"]):
                rows.append({"task": "t2c", "src": line.strip(), "tgt": ex["tgt"], "provider": t["id"],
                             "model": t["model"], "date": date.today().isoformat(), "license": t["license_note"],
                             "training_allowed": True, "origin": "api", "user_rating": 0})
        time.sleep(60.0 / max(1, t["limits"]["rpm"]))
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"teachers": len(teachers), "rows": len(rows), "calls": used}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=200)
    ap.add_argument("--per-prompt", type=int, default=4)
    ap.add_argument("--max-per-provider", type=int, default=150)
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    a = ap.parse_args(argv)
    print(json.dumps(collect(a.prompts, a.per_prompt, Path(a.out), a.seed, a.max_per_provider), ensure_ascii=False))


if __name__ == "__main__":
    main()
