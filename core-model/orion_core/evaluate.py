"""Değerlendirme + yayın kapısı (eval gate).

Değerlendirme, telefona gidecek NİCEMLENMİŞ (int8) model üzerinde yapılır.
Eşikler shared/core-model/gate.json içinde; altında kalan model yayınlanmaz,
uygulama önceki sürümle devam eder.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import SHARED, datagen
from .bpe import EOS, PAD, BPE
from .copy import decode_output
from .export import load_quantized
from .protocol import ProtocolError, canonical, validate_request
from .train import prompt_ids

GATE_PATH = SHARED / "core-model" / "gate.json"
FLOOR_PATH = SHARED / "core-model" / "app-floor.json"


def normalize_cmd(text: str) -> str | None:
    """Model çıktısını doğrulayıcıdan geçirip kanonik tam komuta çevirir; geçersizse None."""
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != {"args", "cmd"}:
            return None
        return canonical(validate_request({"v": 1, "id": "g", "cmd": obj["cmd"], "args": obj["args"]}))
    except (json.JSONDecodeError, ProtocolError, TypeError):
        return None


def run_eval(model_bin: Path, tok: Path, n_iid: int = 600, n_ood: int = 400, n_plan: int = 200,
             n_ans: int = 100, batch: int = 64) -> dict:
    model = load_quantized(model_bin)
    bpe = BPE.load(tok)

    confs: dict = {}

    def gen(task: str, srcs: list[str], key: str | None = None) -> list[str]:
        prompts = [prompt_ids(bpe, task, s) for s in srcs]
        outs, conf = model.greedy_batch(prompts, EOS, PAD, max_new=96, batch=batch, with_conf=True)
        if key:
            confs[key] = conf
        return [decode_output(bpe, p[2:-1], o) for p, o in zip(prompts, outs)]

    res: dict = {}
    for split, n, seed in (("test_iid", n_iid, 101), ("test_ood", n_ood, 202), ("legacy_ood", n_ood, 202)):
        exs = datagen.t2c_set(n, split, seed)
        outs = gen("t2c", [e["src"] for e in exs], split)
        ok = valid = 0
        per_cmd: dict = defaultdict(lambda: [0, 0])
        fails = []
        for e, o in zip(exs, outs):
            norm = normalize_cmd(o)
            cmd = json.loads(e["tgt"])["cmd"]
            per_cmd[cmd][1] += 1
            if norm is not None:
                valid += 1
            if norm == e["full"]:
                ok += 1
                per_cmd[cmd][0] += 1
            elif len(fails) < 15:
                fails.append({"src": e["src"], "want": e["tgt"], "got": o})
        hits = [normalize_cmd(o) == e["full"] for e, o in zip(exs, outs)]
        res[split + "_hits"] = hits
        res[split] = {"exact": ok / len(exs), "valid": valid / len(exs), "n": len(exs),
                      "per_cmd": {k: round(v[0] / v[1], 3) for k, v in sorted(per_cmd.items())}, "fails": fails}

    hw_path = SHARED / "core-model" / "handwritten-eval.jsonl"
    if hw_path.exists():
        hw = [json.loads(x) for x in hw_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        outs = gen("t2c", [h["src"] for h in hw], "handwritten")
        full = [canonical(validate_request({"v": 1, "id": "g", **json.loads(h["tgt"])})) for h in hw]
        hits = [normalize_cmd(o) == f for o, f in zip(outs, full)]
        res["handwritten_hits"] = hits
        res["handwritten"] = {"exact": sum(hits) / len(hw), "n": len(hw),
                              "fails": [{"src": h["src"], "want": h["tgt"], "got": o}
                                        for h, o, ok in zip(hw, outs, hits) if not ok][:15]}

    import random
    rng = random.Random(303)
    plans = [datagen.plan_example(rng, "test_iid") for _ in range(n_plan)]
    outs = gen("plan", [p["src"] for p in plans])
    res["plan"] = {"exact": sum(o == p["tgt"] for o, p in zip(outs, plans)) / len(plans), "n": len(plans)}
    ans = [datagen.answer_example(rng, "test_iid") for _ in range(n_ans)]
    outs = gen("ans", [a["src"] for a in ans])
    res["ans"] = {"exact": sum(o == a["tgt"] for o, a in zip(outs, ans)) / len(ans), "n": len(ans)}
    # Ayrı kalibrasyon kümeleri (farklı tohum): eşik test kümelerine bakılarak seçilmez.
    cal: dict = {}
    for split, n, seed in (("test_iid", 1000, 707), ("test_ood", 1000, 909), ("legacy_ood", 1000, 808)):
        exs = datagen.t2c_set(n, split, seed)
        outs = gen("t2c", [e["src"] for e in exs], "cal_" + split)
        cal[split] = list(zip(confs["cal_" + split], [normalize_cmd(o) == e["full"] for e, o in zip(exs, outs)]))
    res["abstain"] = abstention(confs, res, cal)
    res["confirm"] = confirm_band(cal, res["abstain"]["threshold"])
    for k in [k for k in res if k.endswith("_hits")]:
        del res[k]
    return res


def abstention(confs: dict, res: dict, cal: dict, margin: float = 0.7) -> dict:
    """Emin değilse sus: güven eşiği seçimi. Eşik, AYRI kalibrasyon kümelerinde her türün yanlış-iş oranı
    güvenlik tabanı sınırının %70'inin altında kalacak en düşük değerdir (kapsam en büyük olsun diye).
    Telefonda bu eşiğin altı 'anlamadım' sayılır. Raporlanan sayılar test kümelerindendir."""
    floor = json.loads(FLOOR_PATH.read_text(encoding="utf-8"))
    limits = {"test_iid": floor["wrong_actions_iid_max"], "test_ood": floor["wrong_actions_ood_max"],
              "legacy_ood": floor["wrong_actions_legacy_ood_max"]}
    best = 1.0
    # Aday eşikler: kalibrasyon güvenlerinin kendisi (ince ayar), 4 basamağa yuvarlanır (telefona da gider).
    grid = sorted({round(c, 4) for pairs in cal.values() for c, _ in pairs} | {1.0})
    for tau in grid:
        if all(sum(1 for c, h in pairs if c >= tau and not h) / len(pairs) <= margin * limits[k]
               for k, pairs in cal.items()):
            best = tau
            break
    target_precision = None
    out = {"threshold": best, "target_precision": target_precision}
    for k in ("test_iid", "test_ood", "legacy_ood", "handwritten"):
        cs, hs = confs.get(k, []), res.get(k + "_hits", [])
        if not cs:
            continue
        acc = [h for c, h in zip(cs, hs) if c >= best]
        out[k] = {"coverage": round(len(acc) / len(cs), 3),
                  "precision": round(sum(acc) / len(acc), 3) if acc else None,
                  "wrong_actions": round(sum(1 for c, h in zip(cs, hs) if c >= best and not h) / len(cs), 3)}
    return out


DEFAULT_CONFIRM_THRESHOLD = 0.8


def confirm_band(cal: dict, abstain: float, min_accuracy: float = 0.6, min_support: int = 20,
                 default: float = DEFAULT_CONFIRM_THRESHOLD) -> dict:
    """"Emin değilsem sorayım" eşiği: güven [c, abstain) aralığındaki çıktı telefonda çalıştırılmaz, onaya sunulur.
    c, AYRI kalibrasyon kümelerinde (hepsi birlikte) bu aralıktaki doğruluğun ≥ min_accuracy olduğu EN DÜŞÜK değerdir
    (en az min_support örnekle). Uygun değer yoksa varsayılan 0.8. c hiçbir zaman abstain eşiğini geçmez."""
    pairs = sorted((c, h) for ps in cal.values() for c, h in ps)
    best = None
    for tau in sorted({round(c, 4) for c, _ in pairs if c < abstain}):
        band = [h for c, h in pairs if tau <= c < abstain]
        if len(band) >= min_support and sum(band) / len(band) >= min_accuracy:
            best = tau
            break
    thr = best if best is not None else min(default, abstain)
    band = [h for c, h in pairs if thr <= c < abstain]
    return {"threshold": thr, "default_used": best is None, "band_n": len(band),
            "band_accuracy": round(sum(band) / len(band), 3) if band else None,
            "min_accuracy": min_accuracy}


def check_gate(res: dict, gate: dict) -> list[str]:
    fails = []
    checks = {
        "t2c_iid_exact": res["test_iid"]["exact"], "t2c_ood_exact": res["test_ood"]["exact"],
        "t2c_iid_valid": res["test_iid"]["valid"], "plan_exact": res["plan"]["exact"], "ans_exact": res["ans"]["exact"],
    }
    for k, v in checks.items():
        if v < gate[k]:
            fails.append(f"{k}={v:.3f} < {gate[k]}")
    return fails


def check_floor(res: dict, floor: dict) -> list[str]:
    """Uygulamaya girme tabanı: geçerlilik + emin değilse susma sayesinde düşük yanlış-iş oranı."""
    fails = []
    if res["test_iid"]["valid"] < floor["t2c_iid_valid"]:
        fails.append(f"t2c_iid_valid={res['test_iid']['valid']:.3f} < {floor['t2c_iid_valid']}")
    ab = res["abstain"]
    cov = ab.get("test_iid", {}).get("coverage", 0.0)
    if cov < floor["coverage_iid_min"]:
        fails.append(f"test_iid kapsam={cov:.3f} < {floor['coverage_iid_min']} (hep susan model işe yaramaz)")
    for k, key in (("test_iid", "wrong_actions_iid_max"), ("test_ood", "wrong_actions_ood_max"),
                   ("legacy_ood", "wrong_actions_legacy_ood_max")):
        w = ab.get(k, {}).get("wrong_actions")
        if w is not None and w > floor[key]:
            fails.append(f"{k} yanlış-iş={w:.3f} > {floor[key]}")
    return fails


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dışa aktarım klasörü (model.bin + tokenizer.json)")
    ap.add_argument("--gate", action="store_true", help="eşik altındaysa çıkış kodu 1")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.model)
    t0 = time.time()
    kw = dict(n_iid=150, n_ood=100, n_plan=60, n_ans=40) if a.quick else {}
    res = run_eval(d / "model.bin", d / "tokenizer.json", **kw)
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    fails = check_gate(res, gate)
    res["gate"] = {"passed": not fails, "fails": fails, "thresholds": gate}
    res["seconds"] = round(time.time() - t0, 1)
    (d / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("test_iid", "test_ood", "legacy_ood"):
        print(f"{k}: tam eşleşme {res[k]['exact']:.3f}, geçerli {res[k]['valid']:.3f}")
    print(f"plan: {res['plan']['exact']:.3f}  cevap: {res['ans']['exact']:.3f}")
    if "handwritten" in res:
        print(f"elle yazılmış (yalnızca rapor): {res['handwritten']['exact']:.3f} ({res['handwritten']['n']} örnek)")
    cb = res.get("confirm", {})
    print(f"emin değilsem sor: eşik {cb.get('threshold')} (aralık doğruluğu {cb.get('band_accuracy')}, n={cb.get('band_n')})")
    ab = res["abstain"]
    print(f"emin değilse sus: eşik {ab['threshold']}  " + "  ".join(
        f"{k}: kapsam {v['coverage']} doğruluk {v['precision']} yanlış-iş {v['wrong_actions']}" for k, v in ab.items() if isinstance(v, dict)))
    floor_fails = check_floor(res, json.loads(FLOOR_PATH.read_text(encoding="utf-8")))
    res["floor"] = {"passed": not floor_fails, "fails": floor_fails}
    (d / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OLGUNLUK KAPISI: GEÇTİ" if not fails else f"OLGUNLUK KAPISI: KALDI {fails} → uygulamada 'gelişiyor' etiketiyle")
    print("GÜVENLİK TABANI: GEÇTİ" if not floor_fails else f"GÜVENLİK TABANI: KALDI {floor_fails} → uygulamaya girmez")
    if a.gate and floor_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
