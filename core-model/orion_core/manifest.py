"""Dışa aktarılan model için manifest.json: sürüm, boyut, parametre, eğitim adımı, değerlendirme sonucu."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import GATE_PATH, check_gate, run_eval


def write_manifest(export_dir: Path, run_dir: Path | None, evaluate: bool = True) -> dict:
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
    }
    if evaluate:
        res = run_eval(model, export_dir / "tokenizer.json")
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        fails = check_gate(res, gate)
        m["eval"] = {
            "t2c_iid_exact": round(res["test_iid"]["exact"], 4), "t2c_ood_exact": round(res["test_ood"]["exact"], 4),
            "t2c_iid_valid": round(res["test_iid"]["valid"], 4), "plan_exact": round(res["plan"]["exact"], 4),
            "ans_exact": round(res["ans"]["exact"], 4),
            "handwritten_exact": round(res.get("handwritten", {}).get("exact", 0.0), 4),
            "legacy_ood_exact": round(res.get("legacy_ood", {}).get("exact", 0.0), 4),
            "gate_passed": not fails, "gate_fails": fails,
            "maturity": "olgun" if not fails else "gelişiyor",
        }
        # Telefon: bu güvenin altındaki çıktı "anlamadım" sayılır (yanlış iş yapmasın).
        m["abstain_threshold"] = res["abstain"]["threshold"]
        (export_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (export_dir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)
    m = write_manifest(Path(a.dir), Path(a.run) if a.run else None, not a.no_eval)
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
