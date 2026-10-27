# Orion sohbet beyni — Kaggle ekran kartı (GPU) oturumu.
# GitHub iş akışı (kaggle.yml) bu dosyadaki __AYAR__ değerlerini doldurup Kaggle'a gönderir.
# Oturum: kodu halka açık depodan alır → Vikipedi verisini hazırlar → önceki oturumun checkpoint'inden devam eder →
# süre bütçesi kadar eğitir → telefona gidecek int8 modeli dışa aktarır. Çıktı /kaggle/working/out altında.
import glob
import os
import subprocess
import sys
import urllib.request

SIZE = "__SIZE__"
STEPS = "__STEPS__"
BATCH = "__BATCH__"
BUDGET_MIN = "__BUDGET__"
REPO = "https://github.com/Eskay9633/Orion-renme-.git"
CKPT_URL = "https://github.com/Eskay9633/Orion-renme-/releases/download/kaggle-devam-__SIZE__/ckpt.pt"


def sh(*a):
    print("+", " ".join(a), flush=True)
    subprocess.run(a, check=True)


sh("git", "clone", "--depth", "1", REPO, "/kaggle/working/src")
sh(sys.executable, "-m", "pip", "install", "-q", "pyarrow", "regex")
os.chdir("/kaggle/working/src/core-model")
data, out = "/kaggle/tmp/chat-data", "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
prev = "/kaggle/tmp/prev-ckpt.pt"
try:
    urllib.request.urlretrieve(CKPT_URL, prev)
    print("önceki checkpoint indirildi, devam edilecek", flush=True)
except Exception as e:  # ilk oturum
    print("önceki checkpoint yok (ilk oturum):", e, flush=True)
    prev = ""
sh(sys.executable, "-m", "orion_core.chat", "prepare", "--out", data)
args = [sys.executable, "-m", "orion_core.chat", "train", "--data", data, "--out", f"{out}/chat-run", "--size", SIZE,
        "--steps", STEPS, "--batch", BATCH, "--budget-min", BUDGET_MIN, "--lr", "5e-4"]
if prev:
    args += ["--resume-from", prev]
sh(*args)
sh(sys.executable, "-m", "orion_core.chat", "export", "--run", f"{out}/chat-run", "--data", data, "--out", f"{out}/chat-export")
for f in glob.glob(f"{out}/chat-run/*.tmp"):
    os.remove(f)
print("bitti", flush=True)
