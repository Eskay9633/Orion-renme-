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
TOK_URL = "__TOK__"  # önceki eğitimin tokenizer'ı (devam ederken aynısı şart); boşsa ilk eğitim
EXTRA_URL = "__EXTRA__"  # isteğe bağlı çeviri verisi (en son ceviri-* Release'indeki ceviri.jsonl.gz); boşsa yok
# İNCE AYAR (yalnız ek veri varsa): diyalog payı 0.75, taban lr 1.5e-4. lr_at kosinüs takvimi toplam hedef adıma
# göre hesaplanır: 70000→90000 devamında etkin lr ~3e-5'ten ~1.5e-5'e iner (5e-4 tabanla ~1e-4'e SIÇRARDI).
# Ek veri varken her 2000 adımda sınav (40 soru, açgözlü; ekran kartında tahmini <1 dk, doğrulanmadı) yapılır,
# en iyi ağırlıklar best.pt'ye yazılır ve ayrıca chat-export-best olarak dışa aktarılır (erken durdurma).
FT_LR, FT_DIALOG_FRAC, FT_EXAM_EVERY = "1.5e-4", "0.75", "2000"


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
tok = ""
if prev and TOK_URL.startswith("http"):
    try:
        urllib.request.urlretrieve(TOK_URL, "/kaggle/tmp/prev-tokenizer.json")
        tok = "/kaggle/tmp/prev-tokenizer.json"
    except Exception as e:
        print("tokenizer indirilemedi:", e, flush=True)
if prev and not tok:
    # Farklı tokenizer'la devam etmek modeli bozar: sıfırdan başla
    print("UYARI: önceki tokenizer yok; checkpoint kullanılmadan sıfırdan eğitilecek", flush=True)
    prev = ""
extra = ""
if EXTRA_URL.startswith("http"):
    try:
        urllib.request.urlretrieve(EXTRA_URL, "/kaggle/tmp/ceviri.jsonl.gz")
        extra = "/kaggle/tmp/ceviri.jsonl.gz"
        print("çeviri verisi indirildi; ince ayar ayarları kullanılacak:", EXTRA_URL, flush=True)
    except Exception as e:  # veri yoksa eski davranış
        print("çeviri verisi indirilemedi, ek veri olmadan devam:", e, flush=True)
prep = [sys.executable, "-m", "orion_core.chat", "prepare", "--out", data]
if tok:
    prep += ["--tokenizer", tok]
if extra:
    prep += ["--extra-dialogs", extra]
sh(*prep)
lr, dfrac = (FT_LR, FT_DIALOG_FRAC) if extra else ("5e-4", "0.25")
args = [sys.executable, "-m", "orion_core.chat", "train", "--data", data, "--out", f"{out}/chat-run", "--size", SIZE,
        "--steps", STEPS, "--batch", BATCH, "--budget-min", BUDGET_MIN, "--lr", lr, "--dialog-frac", dfrac]
if extra:
    args += ["--exam-every", FT_EXAM_EVERY]
if prev:
    args += ["--resume-from", prev]
sh(*args)
sh(sys.executable, "-m", "orion_core.chat", "export", "--run", f"{out}/chat-run", "--data", data, "--out", f"{out}/chat-export")
if os.path.exists(f"{out}/chat-run/best.pt"):
    sh(sys.executable, "-m", "orion_core.chat", "export", "--run", f"{out}/chat-run", "--data", data,
       "--out", f"{out}/chat-export-best", "--ckpt", "best.pt")
for f in glob.glob(f"{out}/chat-run/*.tmp"):
    os.remove(f)
print("bitti", flush=True)
