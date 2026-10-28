"""Gelişim deposundan (halka açık, Eskay9633/Orion-renme-) en yeni modelleri içe aktarır.

İşlem deposunun CI'ında çalışır:
  - komut-*  Release → shared/core-model/  (güvenlik tabanını geçmeli ve elle yazılmış cümlelerde gerilememeli)
  - sohbet-* Release → shared/chat-model/  (shared/chat-model-floor.json tabanını geçmeli)
Değişiklik olursa çıktı: changed=true (iş akışı sürümü artırıp APK'yı derletir).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from . import SHARED

REPO = "Eskay9633/Orion-renme-"
KOMUT_FILES = ("model.bin", "tokenizer.json", "parity.json", "manifest.json")
CHAT_FILES = ("model.bin", "tokenizer.json", "parity.json", "manifest.json", "samples.json")


def _get(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "orion-importer",
                                               **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def latest(prefix: str, token: str | None) -> dict | None:
    rels = json.loads(_get(f"https://api.github.com/repos/{REPO}/releases?per_page=50", token))
    rels = [r for r in rels if r["tag_name"].startswith(prefix) and not r.get("draft")]
    return max(rels, key=lambda r: r["created_at"]) if rels else None


def download(rel: dict, names: tuple[str, ...], dest: Path, token: str | None) -> None:
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    for n in names:
        if n in assets:
            (dest / n).write_bytes(_get(assets[n], None))


def komut_better(new: dict, cur: dict | None) -> tuple[bool, str]:
    """Yeni komut modeli alınır mı? Güvenlik tabanı ayrıca evaluate ile sınanır; burada gerileme denetimi."""
    if cur is None or "eval" not in cur:
        return True, "mevcut model yok"
    if new.get("sha256") == cur.get("sha256"):
        return False, "aynı model"
    n, c = new.get("eval", {}), cur["eval"]
    if n.get("handwritten_exact", 0) + 1e-9 < c.get("handwritten_exact", 0):
        return False, f"elle yazılmış cümlelerde geriledi ({n.get('handwritten_exact')} < {c.get('handwritten_exact')})"
    if n.get("t2c_ood_exact", 0) + 0.02 < c.get("t2c_ood_exact", 0):
        return False, "yeni cümle yapılarında geriledi"
    return True, "daha iyi ya da eşit"


def chat_ok(m: dict, floor: dict, cur: dict | None) -> tuple[bool, str]:
    ppl = m.get("val_ppl_int8")
    if ppl is None or ppl != ppl or ppl > floor["val_ppl_max"]:
        return False, f"şaşkınlık {ppl} > {floor['val_ppl_max']} (henüz yeterince öğrenmedi)"
    if floor.get("persona_ok") and not m.get("persona_ok"):
        return False, "kişilik testi geçmedi (kendini Orion olarak tanıtmıyor)"
    if "qa_min" in floor and (m.get("qa_score") or 0) < floor["qa_min"]:
        return False, f"sınav sorularında doğru cevap {m.get('qa_score')} < {floor['qa_min']} (soruya göre cevap vermiyor)"
    if cur and cur.get("sha256") == m.get("sha256"):
        return False, "aynı model"
    if cur and (cur.get("val_ppl_int8") or 1e9) < ppl:
        return False, "mevcut sohbet modelinden kötü"
    return True, "tabanı geçti"


def run(token: str | None, evaluate_gate) -> dict:
    report: dict = {"changed": False}
    core_dir, chat_dir = SHARED / "core-model", SHARED / "chat-model"
    rel = latest("komut-", token)
    if rel:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            download(rel, KOMUT_FILES, t, token)
            new = json.loads((t / "manifest.json").read_text())
            cur = json.loads((core_dir / "manifest.json").read_text()) if (core_dir / "manifest.json").exists() else None
            ok, why = komut_better(new, cur)
            if ok and not evaluate_gate(t):
                ok, why = False, "güvenlik tabanını bu depoda geçemedi"
            report["komut"] = {"tag": rel["tag_name"], "alındı": ok, "neden": why}
            if ok:
                for n in KOMUT_FILES:
                    shutil.copy(t / n, core_dir / n)
                report["changed"] = True
    rel = latest("sohbet-", token)
    if rel:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            download(rel, CHAT_FILES, t, token)
            m = json.loads((t / "manifest.json").read_text())
            floor = json.loads((SHARED / "chat-model-floor.json").read_text())
            cur = json.loads((chat_dir / "manifest.json").read_text()) if (chat_dir / "manifest.json").exists() else None
            ok, why = chat_ok(m, floor, cur)
            report["sohbet"] = {"tag": rel["tag_name"], "alındı": ok, "neden": why, "ppl": m.get("val_ppl_int8")}
            if ok:
                chat_dir.mkdir(exist_ok=True)
                for n in CHAT_FILES:
                    if (t / n).exists():
                        shutil.copy(t / n, chat_dir / n)
                report["changed"] = True
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="rapor json")
    a = ap.parse_args(argv)

    def gate(d: Path) -> bool:
        from .evaluate import FLOOR_PATH, check_floor, run_eval
        res = run_eval(d / "model.bin", d / "tokenizer.json")
        fails = check_floor(res, json.loads(FLOOR_PATH.read_text(encoding="utf-8")))
        print("taban:", "geçti" if not fails else fails, flush=True)
        return not fails

    rep = run(os.environ.get("GH_TOKEN"), gate)
    txt = json.dumps(rep, ensure_ascii=False, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"changed={'true' if rep['changed'] else 'false'}\n")


if __name__ == "__main__":
    main()
