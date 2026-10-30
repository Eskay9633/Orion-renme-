# Kaggle oturum günlüğünden önemli satırları yazdırır (ORION işaretli, hata, çeviri/ek diyalog, checkpoint).
# Kullanım: python kaggle/gunluk.py <kernels output klasörü>
import glob
import json
import sys

KEYS = ("ORION", "Traceback", "Error", "error", "ek diyalog", "çeviri", "checkpoint", "tokenizer", "Killed", "+ ")
root = sys.argv[1] if len(sys.argv) > 1 else "kout"
files = glob.glob(f"{root}/*.log")
if not files:
    print("kaggle-günlük: günlük dosyası yok")
for f in files:
    try:
        rows = json.load(open(f, encoding="utf-8"))
        text = "".join(r.get("data", "") for r in rows if isinstance(r, dict))
    except Exception:
        text = open(f, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    for line in lines:
        if any(k in line for k in KEYS):
            print("kaggle-günlük:", line[:300])
    print("kaggle-günlük: --- son 25 satır ---")
    for line in lines[-25:]:
        print("kaggle-günlük:", line[:300])
