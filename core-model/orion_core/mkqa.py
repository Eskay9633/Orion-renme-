"""MKQA (Türkçe) bilgi soruları → Orion'un sesiyle kısa cevaplı diyaloglar.

Kaynak: Apple MKQA (Longpre ve ark. 2021), github.com/apple/ml-mkqa, dataset/mkqa.jsonl.gz.
Lisans: CC BY-SA 3.0 (doğrulandı: depo README "License" bölümü). Ayrıntı ve atıf: shared/core-model/DIS-VERI.md.
İndirme sabit commit adresinden yapılır ve sha256 ile doğrulanır; uymazsa ya da indirilemezse iş DURUR (sessiz atlama
yok; Kaggle'da internet var).
Dönüşüm (bilgi UYDURULMAZ; cevapta yalnızca verideki değer geçer):
- entity/short_phrase → "X." / "Bildiğim kadarıyla X." …; number(_with_unit) → ".0" atılır, ondalık virgül;
  date → "YYYY-AA-GG" Türkçe ay adıyla yazılır; binary → evet/hayır cümlesi
- zamana bağlı sorular ("en son", "şu an", "gelecek"…) → cevaba "bu bilgi değişmiş olabilir" notu (veri 2018 dönemi)
- long_answer atılır; unanswerable → en çok UNANSWERABLE_MAX tanesi dürüst "bilmiyorum, araştırayım" cevabı alır
- cevaplanabilir sorulardan N_EVAL (1000) tanesi değerlendirmeye ayrılır (yalnızca rapor; eğitime girmez)
- sınav ayıklaması: dialogs.clean_pair (QA_TESTS + bilişsel sınav) + near_exam
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

MKQA_COMMIT = "f964fe1bac4d580cee3d4928572df1f676f8e656"
MKQA_URL = f"https://raw.githubusercontent.com/apple/ml-mkqa/{MKQA_COMMIT}/dataset/mkqa.jsonl.gz"
MKQA_SHA256 = "f77ac6bf6306e98816c5c047880db71a3c67d2f1491b169896d63e9bb14ef3d3"
LICENSE = "cc-by-sa-3.0"
SOURCE = "apple/ml-mkqa (MKQA, CC BY-SA 3.0)"
N_EVAL = 1000
UNANSWERABLE_MAX = 400
SEED = 2021

MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
# değer cümle ortasında büyük harfle başlayabilir (özel ad mı bilinmez): hep iki noktadan sonra ya da başta
ANSWER_T = ["{x}.", "Bildiğim kadarıyla: {x}.", "Cevap: {x}.", "Bence cevap şu: {x}.", "Aklımdaki cevap: {x}.",
            "Bildiğim kadarıyla cevap: {x}."]
YES_T = ["Evet, bildiğim kadarıyla öyle.", "Bildiğim kadarıyla evet."]
NO_T = ["Hayır, bildiğim kadarıyla öyle değil.", "Bildiğim kadarıyla hayır."]
UNSURE_T = ["Bundan emin değilim; uydurmak istemem. İstersen araştırayım.",
            "Bunu bilmiyorum. İstersen araştırıp bulduklarımı anlatayım.",
            "Buna kesin bir cevap veremem; yanlış bilgi vermektense araştırmayı tercih ederim. Araştırayım mı?"]
TIME_TAIL = ["Bu bilgi değişmiş olabilir; istersen güncelini araştırayım.",
             "Bilgim biraz eski olabilir, istersen güncelini kontrol edelim."]
TIME_WORDS = re.compile(r"\b(en son|şu an|şuan|şimdi|şimdiki|gelecek|yeni sezon|güncel|mevcut|bu yıl|bu sene|son sezon|"
                        r"kaç yaşında|hala|hâlâ|ne zaman çıkacak|çıkıyor|geçen sene|geçen yıl)\b", re.I)


def download(cache: Path, url: str | None = None, sha256: str | None = None) -> Path:
    """Dosyayı indirir (önbellekte doğruysa yeniden indirmez) ve sha256'yı doğrular; uymazsa ValueError.
    url yerel bir dosya yolu da olabilir (test/çevrimdışı); hash yine denetlenir."""
    url, sha256 = url or MKQA_URL, sha256 or MKQA_SHA256
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "mkqa.jsonl.gz"
    if not (path.exists() and _sha(path) == sha256):
        src = Path(url)
        if not url.startswith(("http://", "https://")) and src.exists():
            path.write_bytes(src.read_bytes())
        else:
            print(f"MKQA indiriliyor: {url}", flush=True)
            tmp = path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)  # ağ yoksa hata yükselir (iş durur)
            tmp.replace(path)
    got = _sha(path)
    if got != sha256:
        raise ValueError(f"MKQA sha256 uyuşmuyor: beklenen {sha256}, gelen {got}")
    return path


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _num(t: str) -> str:
    """"5.0" → "5", "1.5" → "1,5", "1500000.0" → "1.500.000"."""
    try:
        v = float(t)
    except ValueError:
        return t
    if v == int(v):
        i = int(v)
        return f"{i:,}".replace(",", ".") if abs(i) >= 10000 else str(i)
    return str(v).replace(".", ",")


def _date(t: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return f"{int(m[3])} {MONTHS[int(m[2]) - 1]} {m[1]}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if m:
        return f"{MONTHS[int(m[2]) - 1]} {m[1]}"
    m = re.fullmatch(r"(\d{2})-(\d{2})", t)  # yılsız "AA-GG"
    if m and 1 <= int(m[1]) <= 12:
        return f"{int(m[2])} {MONTHS[int(m[1]) - 1]}"
    return t


def answer_value(answers: list[dict]) -> tuple[str, str] | None:
    """(tür, Türkçe değer) ya da None (long_answer/boş)."""
    if not answers:
        return None
    typ = answers[0].get("type")
    if typ == "unanswerable":
        return ("unanswerable", "")
    if typ == "long_answer":
        return None
    if typ == "binary":
        t = (answers[0].get("text") or "").lower()
        return ("binary", t) if t in ("yes", "no") else None
    vals = []
    for a in answers[:1]:  # birden çok cevap çoğu zaman alternatiftir; birleştirmek çelişki üretir
        if a.get("type") != typ or not (a.get("text") or "").strip():
            continue
        t = a["text"].strip()
        if typ == "number":
            t = _num(t)
        elif typ == "number_with_unit":
            m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s+(\D.*)", t)
            if not m:
                continue  # "55 67.0 Fit" gibi bozuk değerler atlanır
            t = f"{_num(m[1])} {m[2]}"
        elif typ == "date":
            t = _date(t)
        if t not in vals:
            vals.append(t)
    if not vals:
        return None
    x = vals[0]
    if len(x) > 120:
        return None
    return typ, x


def question(q: str, rng: random.Random) -> str:
    q = re.sub(r"\s+", " ", q or "").strip()
    q = q[:1].upper() + q[1:] if q[:1] not in ("i", "ı") else {"i": "İ", "ı": "I"}[q[:1]] + q[1:]
    if not q.endswith("?") and rng.random() < 0.6:
        q += "?"
    return q


def to_pair(q: str, answers: list[dict], rng: random.Random) -> tuple[str, str] | None:
    v = answer_value(answers)
    if v is None:
        return None
    typ, x = v
    if typ == "unanswerable":
        a = rng.choice(UNSURE_T)
    elif typ == "binary":
        a = rng.choice(YES_T if x == "yes" else NO_T)
    else:
        a = rng.choice(ANSWER_T).format(x=x)
        a = {"i": "İ", "ı": "I"}.get(a[:1], a[:1].upper()) + a[1:]
        if TIME_WORDS.search(q):
            a += " " + rng.choice(TIME_TAIL)
    return question(q, rng), a


def load_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                q = (r.get("queries") or {}).get("tr")
                if q:
                    rows.append({"id": r.get("example_id"), "q": q, "answers": (r.get("answers") or {}).get("tr") or []})
    return rows


def _exam_safe(q: str) -> bool:
    from .dialogs import clean_pair, near_exam
    return not near_exam(q) and clean_pair(q, "Bu cevap yeterince uzun ve Türkçe bir cümledir.") is not None


def build(path: Path, n_eval: int = N_EVAL, seed: int = SEED) -> tuple[list[str], list[dict], dict]:
    """(eğitim diyalogları [format_dialog], değerlendirme satırları, istatistik). Belirlenimci."""
    from .dialogs import IDENTITY, format_dialog
    rng = random.Random(seed)
    rows = sorted(load_rows(path), key=lambda r: (str(r["id"]), r["q"]))
    rng.shuffle(rows)
    stats = {"satir": len(rows), "turler": {}, "atilan": {}}
    answerable, unans = [], []
    for r in rows:
        v = answer_value(r["answers"])
        t = (r["answers"][0].get("type") if r["answers"] else "yok")
        stats["turler"][t] = stats["turler"].get(t, 0) + 1
        if v is None:
            stats["atilan"]["uzun_ya_da_bos"] = stats["atilan"].get("uzun_ya_da_bos", 0) + 1
            continue
        if not _exam_safe(r["q"]):
            stats["atilan"]["sinav_ya_da_suzgec"] = stats["atilan"].get("sinav_ya_da_suzgec", 0) + 1
            continue
        (unans if v[0] == "unanswerable" else answerable).append(r)
    eval_rows = [{"id": r["id"], "q": r["q"], "type": answer_value(r["answers"])[0],
                  "golds": sorted({g for a in r["answers"] for g in [a.get("text") or ""] + list(a.get("aliases") or [])
                                   if g})} for r in answerable[:n_eval]]
    texts = []
    for r in answerable[n_eval:] + unans[:UNANSWERABLE_MAX]:
        p = to_pair(r["q"], r["answers"], rng)
        # clean_pair'in Türkçe oranı denetimi kısa özel ad cevaplarını (ör. "Cevap: Percy Sledge.") atardı;
        # soru zaten _exam_safe'ten geçti, burada yalnızca uzunluk ve başka model kimliği denetlenir.
        c = p if p and 3 <= len(p[0]) <= 600 and not IDENTITY.search(p[0] + " " + p[1]) else None
        if c is None:
            stats["atilan"]["donusum"] = stats["atilan"].get("donusum", 0) + 1
            continue
        texts.append(format_dialog(*c))
    stats.update(egitim=len(texts), degerlendirme=len(eval_rows), cevapsiz_egitim=min(len(unans), UNANSWERABLE_MAX),
                 kaynak=SOURCE, lisans=LICENSE)
    return texts, eval_rows, stats


def eval_score(replies: list[str], rows: list[dict]) -> float:
    """Yalnızca rapor: cevapta altın değerlerden biri (ya da evet/hayır) geçiyor mu?"""
    from .dialogs import fold
    ok = 0
    for a, r in zip(replies, rows):
        fa = f" {fold(a)} "
        golds = r["golds"]
        if r["type"] == "binary":
            golds = ["evet"] if "yes" in golds else ["hayır"]
        ok += any(f" {fold(g)} " in fa for g in golds if fold(g))
    return ok / max(1, len(rows))
