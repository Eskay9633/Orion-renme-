"""Dışa aktarılan model için manifest.json: sürüm, boyut, parametre, eğitim adımı, değerlendirme sonucu."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import GATE_PATH, check_gate, run_eval
from .massive import DATA_SOURCES


def write_manifest(export_dir: Path, run_dir: Path | None, evaluate: bool = True, ext_dir: Path | None = None) -> dict:
    model = export_dir / "model.bin"
    raw = model.read_bytes()
    head_len = int.from_bytes(raw[4:8], "little")
    cfg = json.loads(raw[8:8 + head_len])["config"]
    status = json.loads((run_dir / "status.json").read_text()) if run_dir and (run_dir / "status.json").exists() else {}
    now = datetime.now(timezone.utc)
    m = {
        "version": now.strftime("%Y.%m.%d-%H%M"),
        "created": now.isoformat(timespec="seconds"),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "config": cfg,
        "params_approx": cfg["vocab_size"] * cfg["d_model"] + cfg["ctx"] * cfg["d_model"]
        + cfg["n_layer"] * (12 * cfg["d_model"] ** 2),
        "train_steps": status.get("step"),
        "train_loss": status.get("loss"),
        # Eğitim/değerlendirme veri kaynakları ve lisansları (atıf: shared/core-model/DIS-VERI.md)
        "data_sources": [{"id": "orion-datagen", "license": "Orion (programla üretilmiş)", "use": "train + test kümeleri"}],
    }
    if status.get("ext_rows"):
        m["data_sources"].append({**DATA_SOURCES[0], "train_rows": status["ext_rows"], "train_frac": status.get("ext_frac")})
    if evaluate:
        res = run_eval(model, export_dir / "tokenizer.json", ext_dir=ext_dir)
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        fails = check_gate(res, gate)
        m["eval"] = {
            "t2c_iid_exact": round(res["test_iid"]["exact"], 4), "t2c_ood_exact": round(res["test_ood"]["exact"], 4),
            "t2c_iid_valid": round(res["test_iid"]["valid"], 4), "plan_exact": round(res["plan"]["exact"], 4),
            "ans_exact": round(res["ans"]["exact"], 4),
            "handwritten_exact": round(res.get("handwritten", {}).get("exact", 0.0), 4),
            # Hangi elle yazılmış kümede ölçüldü: içe aktarıcı yalnızca AYNI kümedeki sayıları karşılaştırır.
            "handwritten_n": res.get("handwritten", {}).get("n", 0),
            "handwritten_set_sha256": res.get("handwritten", {}).get("set_sha256"),
            "legacy_ood_exact": round(res.get("legacy_ood", {}).get("exact", 0.0), 4),
            # Gerçek Türkçe cümle kümeleri: yalnızca rapor (kapıya girmez); hazırlanmadıysa yok.
            **{f"{k}_{f}": round(res[k][f], 4) for k in ("massive_tr", "xsid_tr") if "exact" in res.get(k, {})
               for f in ("exact", "wrong_actions")},
            **{f"{k}_n": res[k]["n"] for k in ("massive_tr", "xsid_tr") if "exact" in res.get(k, {})},
            **{f"{k}_set_sha256": res[k]["set_sha256"] for k in ("massive_tr", "xsid_tr") if "exact" in res.get(k, {})},
            "gate_passed": not fails, "gate_fails": fails,
            "maturity": "olgun" if not fails else "gelişiyor",
        }
        for src, key in ((DATA_SOURCES[0], "massive_tr"), (DATA_SOURCES[1], "xsid_tr")):
            if "exact" in res.get(key, {}) and all(d["id"] != src["id"] for d in m["data_sources"]):
                m["data_sources"].append(src)
        # Telefon: bu güvenin altındaki çıktı "anlamadım" sayılır (yanlış iş yapmasın).
        m["abstain_threshold"] = res["abstain"]["threshold"]
        # Telefon: güven [confirm_threshold, abstain_threshold) → yan etkili işte çalıştırmadan önce sorulur.
        m["confirm_threshold"] = res.get("confirm", {}).get("threshold", 0.8)
        (export_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (export_dir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--ext-data", help="massive.py prepare çıktısı (yalnızca rapor kümeleri)")
    a = ap.parse_args(argv)
    m = write_manifest(Path(a.dir), Path(a.run) if a.run else None, not a.no_eval,
                       Path(a.ext_data) if a.ext_data else None)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
