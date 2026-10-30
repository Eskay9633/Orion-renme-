"""Gelişim deposundan (halka açık, Eskay9633/orion-egitim-kampi) en yeni modelleri içe aktarır.

İşlem deposunun CI'ında çalışır:
  - komut-*  Release → shared/core-model/  (güvenlik tabanını geçmeli ve AYNI elle yazılmış kümede gerilememeli)
  - sohbet-* Release → shared/chat-model/  (shared/chat-model-floor.json tabanını geçmeli; sınav ve kişilik puanı
    manifestten değil, aday indirilip burada chat.evaluate_release ile yeniden ölçülür. "hold": true iken alınmaz)
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

REPO = "Eskay9633/orion-egitim-kampi"
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


def recent(prefix: str, token: str | None, n: int = 8) -> list[dict]:
    """En yeniden eskiye son n Release (taslaklar hariç)."""
    rels = json.loads(_get(f"https://api.github.com/repos/{REPO}/releases?per_page=50", token))
    rels = [r for r in rels if r["tag_name"].startswith(prefix) and not r.get("draft")]
    return sorted(rels, key=lambda r: r["created_at"], reverse=True)[:n]


def pick_chat(cands: list[tuple[str, dict]], floor: dict, cur: dict | None,
              local_eval=None, max_eval: int = 3) -> tuple[int | None, list[dict]]:
    """Sohbet adayları (en yeniden eskiye): tabanı geçen ilk aday seçilir. Birden çok eğitim hattı (Kaggle GPU,
    GitHub CPU) yayın yaptığı için yalnızca en yeniye bakmak, iyi bir modelin sonradan gelen zayıf biri yüzünden
    hiç alınmamasına yol açar.

    local_eval(tag) → {"qa_score", "persona_ok", ...}: verilirse ucuz denetimi (şaşkınlık, aynı model) geçen aday
    indirilip BU depoda sabit istem + güncel sınavla yeniden değerlendirilir; karar yerel puanlarla verilir
    (manifest puanları yalnızca karşılaştırma için raporda kalır). En çok max_eval aday indirilir."""
    tried = []
    evals = 0
    for i, (tag, m) in enumerate(cands):
        row = {"tag": tag, "ppl": m.get("val_ppl_int8"), "manifest_qa": m.get("qa_score"),
               "manifest_kisilik": m.get("persona_ok")}
        ok, why = chat_cheap(m, floor, cur)
        if ok and local_eval is not None:
            if evals >= max_eval:
                ok, why = False, f"değerlendirme sınırı ({max_eval} indirme) doldu"
            else:
                evals += 1
                try:
                    loc = local_eval(tag)
                except Exception as e:  # noqa: BLE001 — yerel değerlendirme yapılamazsa manifeste güvenilmez
                    loc = None
                    ok, why = False, f"yerel değerlendirme yapılamadı: {e}"
                if loc is not None:
                    row["yerel_qa"], row["yerel_kisilik"] = loc.get("qa_score"), loc.get("persona_ok")
                    ok, why = chat_ok({**m, "qa_score": loc.get("qa_score"), "persona_ok": loc.get("persona_ok")}, floor, cur)
        elif ok:
            ok, why = chat_ok(m, floor, cur)
        row.update({"alındı": ok, "neden": why})
        tried.append(row)
        if ok:
            return i, tried
    return None, tried


def download(rel: dict, names: tuple[str, ...], dest: Path, token: str | None) -> None:
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    for n in names:
        if n in assets:
            (dest / n).write_bytes(_get(assets[n], None))


def same_handwritten_set(new: dict, cur: dict | None) -> bool:
    """İki manifestin elle yazılmış puanı AYNI kümede mi ölçüldü? (Küme kimliği yoksa: bilinmiyor = hayır.)"""
    a = (new or {}).get("eval", {}).get("handwritten_set_sha256")
    b = (cur or {}).get("eval", {}).get("handwritten_set_sha256")
    return bool(a) and a == b


def komut_better(new: dict, cur: dict | None, local_hw: tuple[float, float] | None = None,
                 local_hw_error: str | None = None) -> tuple[bool, str]:
    """Yeni komut modeli alınır mı? Güvenlik tabanı ayrıca evaluate ile sınanır; burada gerileme denetimi.

    Elle yazılmış cümlelerde gerileme kuralı toleranssız katıdır ve yalnızca AYNI kümede ölçülmüş iki sayıyla uygulanır:
      - local_hw = (yeni, mevcut) verilmişse (içe aktarıcı İKİ modeli de bu depodaki güncel kümede yeniden ölçtü)
        bunlar karşılaştırılır — kümeler aynı olsa da tercih edilen yol budur (109 cümle, ucuz);
      - yerel ölçüm yoksa ve iki manifest aynı küme kimliğini (handwritten_set_sha256) taşıyorsa manifest sayıları;
      - ikisi de yoksa KAPALI kalır: model alınmaz ("elle yazılmış küme ölçülemedi: <hata>"). Ölçülemeyen bir
        gerileme "geçti" sayılmaz.
    """
    if cur is None or "eval" not in cur:
        return True, "mevcut model yok"
    if new.get("sha256") == cur.get("sha256"):
        return False, "aynı model"
    n, c = new.get("eval", {}), cur["eval"]
    if local_hw is not None:
        ln, lc = local_hw
        if ln + 1e-9 < lc:
            return False, f"elle yazılmış cümlelerde geriledi (bu depoda aynı kümede: {ln:.4f} < {lc:.4f})"
        hw_note = f"; elle yazılmış küme bu depoda yeniden ölçüldü ({ln:.4f} ≥ {lc:.4f})"
    elif same_handwritten_set(new, cur):
        if n.get("handwritten_exact", 0) + 1e-9 < c.get("handwritten_exact", 0):
            return False, f"elle yazılmış cümlelerde geriledi ({n.get('handwritten_exact')} < {c.get('handwritten_exact')})"
        hw_note = "; elle yazılmış küme aynı, manifest sayıları karşılaştırıldı"
        if local_hw_error:
            hw_note += f" (yerel ölçüm yapılamadı: {local_hw_error})"
    else:
        return False, f"elle yazılmış küme ölçülemedi: {local_hw_error or 'kümeler farklı ve bu depoda yeniden ölçülmedi'}"
    if n.get("t2c_ood_exact", 0) + 0.02 < c.get("t2c_ood_exact", 0):
        return False, "yeni cümle yapılarında geriledi"
    return True, "daha iyi ya da eşit" + hw_note


def chat_cheap(m: dict, floor: dict, cur: dict | None) -> tuple[bool, str]:
    """İndirmeden yapılan ucuz denetim (manifest): şaşkınlık tabanı, aynı model, mevcuttan kötü."""
    ppl = m.get("val_ppl_int8")
    if ppl is None or ppl != ppl or ppl > floor["val_ppl_max"]:
        return False, f"şaşkınlık {ppl} > {floor['val_ppl_max']} (henüz yeterince öğrenmedi)"
    if cur and cur.get("sha256") == m.get("sha256"):
        return False, "aynı model"
    if cur and (cur.get("val_ppl_int8") or 1e9) < ppl:
        return False, "mevcut sohbet modelinden kötü"
    return True, "şaşkınlık tabanı geçti"


def chat_ok(m: dict, floor: dict, cur: dict | None) -> tuple[bool, str]:
    ok, why = chat_cheap(m, floor, cur)
    if not ok:
        return ok, why
    if floor.get("persona_ok") and not m.get("persona_ok"):
        return False, "kişilik testi geçmedi (kendini Orion olarak tanıtmıyor)"
    if "qa_min" in floor and (m.get("qa_score") or 0) < floor["qa_min"]:
        return False, f"sınav sorularında doğru cevap {m.get('qa_score')} < {floor['qa_min']} (soruya göre cevap vermiyor)"
    return True, "tabanı geçti"


def _local_eval(d: Path) -> dict:
    """Sohbet beynini bu depoda değerlendirir (sabit istem + güncel sınav). torch gerektirir; tembel içe aktarma."""
    from .chat import evaluate_release
    return evaluate_release(d)


def _local_handwritten(d: Path) -> float:
    """Bir komut modelini bu depodaki güncel elle yazılmış kümede ölçer (torch gerektirir; tembel içe aktarma)."""
    from .evaluate import handwritten_eval
    return handwritten_eval(d / "model.bin", d / "tokenizer.json")["exact"]


def run(token: str | None, evaluate_gate, measure_handwritten=None) -> dict:
    report: dict = {"changed": False}
    core_dir, chat_dir = SHARED / "core-model", SHARED / "chat-model"
    measure = measure_handwritten or _local_handwritten
    rel = latest("komut-", token)
    if rel:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            download(rel, KOMUT_FILES, t, token)
            new = json.loads((t / "manifest.json").read_text())
            cur = json.loads((core_dir / "manifest.json").read_text()) if (core_dir / "manifest.json").exists() else None
            local_hw, local_err = None, None
            if cur and "eval" in cur and new.get("sha256") != cur.get("sha256"):
                # İki model de burada, güncel elle yazılmış kümede ölçülür (kümeler aynı olsa da: ucuz ve kesin).
                try:
                    if not (core_dir / "model.bin").exists():
                        raise FileNotFoundError("mevcut model dosyası (model.bin) bu depoda yok")
                    local_hw = (measure(t), measure(core_dir))
                    report["elle_yerel"] = {"yeni": round(local_hw[0], 4), "mevcut": round(local_hw[1], 4)}
                except Exception as e:  # noqa: BLE001 — ölçülemezse: aynı kümede manifest, değilse model ALINMAZ
                    local_err = str(e) or type(e).__name__
                    report["elle_yerel"] = {"hata": local_err}
            ok, why = komut_better(new, cur, local_hw, local_err)
            if ok and not evaluate_gate(t):
                ok, why = False, "güvenlik tabanını bu depoda geçemedi"
            report["komut"] = {"tag": rel["tag_name"], "alındı": ok, "neden": why}
            if ok:
                for n in KOMUT_FILES:
                    shutil.copy(t / n, core_dir / n)
                report["changed"] = True
    floor = json.loads((SHARED / "chat-model-floor.json").read_text())
    cur = json.loads((chat_dir / "manifest.json").read_text()) if (chat_dir / "manifest.json").exists() else None
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        rels, cands = recent("sohbet-", token), []
        for r in rels:
            d = t / r["tag_name"]
            d.mkdir()
            download(r, ("manifest.json",), d, token)
            if (d / "manifest.json").exists():
                cands.append((r["tag_name"], json.loads((d / "manifest.json").read_text())))

        def evaluate(tag: str) -> dict:
            rel = next(r for r in rels if r["tag_name"] == tag)
            d = t / tag
            download(rel, ("model.bin", "tokenizer.json"), d, token)
            return _local_eval(d)

        i, tried = pick_chat(cands, floor, cur, local_eval=evaluate)
        if tried:
            report["sohbet_adaylar"] = tried
            report["sohbet"] = dict(tried[i] if i is not None else tried[0])
        if i is not None and floor.get("hold"):
            # Beklemede: değerlendirildi ve raporlandı ama uygulamaya alınmaz (yeni APK yok); kullanıcı onayıyla açılır.
            report["sohbet"].update({"alındı": False, "neden": "beklemede (kullanıcı onayı)", "tabanı_geçti": True,
                                    "beklemede": True})
        elif i is not None:
            rel = next(r for r in rels if r["tag_name"] == cands[i][0])
            d = t / rel["tag_name"]
            download(rel, tuple(n for n in CHAT_FILES if not (d / n).exists()), d, token)
            chat_dir.mkdir(exist_ok=True)
            for n in CHAT_FILES:
                if (d / n).exists():
                    shutil.copy(d / n, chat_dir / n)
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
