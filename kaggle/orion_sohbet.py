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
REPO = "https://github.com/Eskay9633/orion-egitim-kampi.git"
CKPT_URL = "https://github.com/Eskay9633/orion-egitim-kampi/releases/download/kaggle-devam-__SIZE__/ckpt.pt"
TOK_URL = "__TOK__"  # önceki eğitimin tokenizer'ı (devam ederken aynısı şart); boşsa ilk eğitim
EXTRA_URL = "__EXTRA__"  # isteğe bağlı çeviri verisi (en son ceviri-* Release'indeki ceviri.jsonl.gz); boşsa yok
# İNCE AYAR (yalnız ek veri varsa): diyalog payı 0.75, taban lr 1.5e-4. lr_at kosinüs takvimi toplam hedef adıma
# göre hesaplanır: 70000→90000 devamında etkin lr ~3e-5'ten ~1.5e-5'e iner (5e-4 tabanla ~1e-4'e SIÇRARDI).
# Ek veri varken her 2000 adımda sınav (40 soru, açgözlü; ekran kartında tahmini <1 dk, doğrulanmadı) yapılır,
# en iyi ağırlıklar best.pt'ye yazılır ve ayrıca chat-export-best olarak dışa aktarılır (erken durdurma).
FT_LR, FT_DIALOG_FRAC, FT_EXAM_EVERY = "1.5e-4", "0.85", "2000"  # Tur 22: Vikipedi payı 0.25 → 0.15


def sh(*a):
    print("+", " ".join(a), flush=True)
    subprocess.run(a, check=True)


sh("git", "clone", "--depth", "1", REPO, "/kaggle/working/src")
sh(sys.executable, "-m", "pip", "install", "-q", "pyarrow", "regex")
os.chdir("/kaggle/working/src/core-model")
data, out = "/kaggle/tmp/chat-data", "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
# Kaggle'ın yeni görüntüsünde /kaggle/tmp hazır gelmiyor (29 Eki: tüm indirmeler FileNotFoundError verdi).
os.makedirs("/kaggle/tmp", exist_ok=True)
prev = "/kaggle/tmp/prev-ckpt.pt"
try:
    urllib.request.urlretrieve(CKPT_URL, prev)
    print("ORION: önceki checkpoint indirildi, devam edilecek", flush=True)
except Exception as e:
    # Devam noktası varken (tokenizer adresi verilmiş) checkpoint inemezse sıfırdan eğitmek beyni bozar: dur.
    if TOK_URL.startswith("http"):
        raise SystemExit(f"ORION HATA: devam checkpoint'i indirilemedi: {e!r}")
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
    # Çeviri verisi istendiyse SESSİZCE atlanmaz (76000 oturumu indirme hatasıyla eski veriyle eğitilmişti):
    # birkaç deneme, olmazsa oturum hata verir; zamanlanmış iş akışı günlüğü gösterir, kendiliğinden yeniden başlatmaz.
    import time
    err = None
    for attempt in range(1, 6):
        try:
            # Yedekli indirme: urllib → curl → wget (hangisi Kaggle ağında çalışırsa)
            dst = "/kaggle/tmp/ceviri.jsonl.gz"
            try:
                urllib.request.urlretrieve(EXTRA_URL, dst)
            except Exception as e1:
                print(f"ORION: urllib indiremedi: {e1!r}; curl deneniyor", flush=True)
                r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", dst, EXTRA_URL], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"ORION: curl indiremedi ({r.returncode}): {r.stderr[-300:]}; wget deneniyor", flush=True)
                    r = subprocess.run(["wget", "-q", "-O", dst, EXTRA_URL], capture_output=True, text=True)
                    if r.returncode != 0:
                        raise IOError(f"urllib/curl/wget başarısız: {e1!r}; wget {r.returncode} {r.stderr[-200:]}")
            size = os.path.getsize("/kaggle/tmp/ceviri.jsonl.gz")
            if size < 1000:
                raise IOError(f"dosya çok küçük ({size} bayt)")
            extra = "/kaggle/tmp/ceviri.jsonl.gz"
            print(f"ORION: çeviri verisi indirildi ({size} bayt); ince ayar ayarları kullanılacak:", EXTRA_URL, flush=True)
            break
        except Exception as e:
            err = e
            print(f"ORION: çeviri verisi indirilemedi (deneme {attempt}/5): {e!r}", flush=True)
            time.sleep(10 * attempt)
    if not extra:
        raise SystemExit(f"ORION HATA: çeviri verisi istendi ama indirilemedi: {err!r}")
# MKQA Türkçe bilgi soruları (CC BY-SA 3.0) chat.prepare içinde sabit commit adresinden sha256 doğrulamasıyla iner;
# inemezse ya da hash uymazsa prepare hata verir (sessiz atlama yok). Ek argüman gerekmez.
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
