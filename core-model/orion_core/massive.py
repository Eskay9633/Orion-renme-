"""Gerçek Türkçe cümleler: Amazon MASSIVE 1.1 (tr-TR) eğitim + değerlendirme, xSID 0.6 (tr) bağımsız değerlendirme.

Lisanslar (doğrulandı, dosyanın kendisinden okundu):
  - MASSIVE 1.1: CC BY 4.0 (tarball içindeki 1.1/LICENSE; github.com/alexa/massive NOTICE.md). © Amazon.com, Inc.
    SLURP (CC BY 4.0) üzerine kuruludur. Ayrıntı: shared/core-model/DIS-VERI.md
  - xSID 0.6: CC BY-SA 4.0 (github.com/mainlp/xsid LICENSE). Yalnızca DEĞERLENDİRME için kullanılır (eğitime girmez).

Akış:
  prepare  indir (adres + sha256 sabitli) ya da yerel dosyayı doğrula → tr-TR satırlarını ayrıştır → kalite süzgeci →
           Orion çiftlerine çevir → sızıntı süzgeci → work/ext/{massive-train,massive_tr,xsid_tr}.jsonl + stats.json
  train.py --ext-data work/ext   eğitim akışına (yalnızca eğitim üreticisi yolu) massive-train.jsonl karışır
  evaluate                       massive_tr ve xsid_tr yalnızca RAPOR (kapı eşiklerine girmez)

Çeviri kuralları (uydurma argüman yok; aşağıdakilerden biri tutmazsa satır ATLANIR, sebebi stats.json'a yazılır):
  - Orion'da karşılığı olan niyetler yalnızca argümanlar cümleden birebir çıkarılabiliyorsa olumlu örnek olur
    (zaman yuvaları Orion 'when' biçimine çevrilir: d+1@09:00, fri@20:00, 03-11@09:00, +10m; çözülemeyen ya da
    sabah/akşam belirsiz saat → atla). Yuvalar dışında kalan sözcükler küçük bir "bağlaç/fiil" kümesinde olmalı
    (ör. "bana", "hatırlat", "için"); başka anlamlı sözcük kalırsa (kişi, yer, "önce" ...) argüman eksik kalır → atla.
  - Orion'da komutu OLMAYAN niyetler (ışık, ses, hava durumu, silme/iptal, sipariş, taksi ...) "kapsam dışı" olumsuz
    örnek olur: çeviricinin bilinmeyen istek için zaten kullandığı biçim → {"cmd":"ask","args":{"question": cümle}}
    (Kotlin Translator 6. adım: hiçbir komuta uymayan istek, kibarlık sözcükleri atılıp sohbete gider).
  - Karşılığı yaklaşık olan niyetler (müzik/radyo/podcast çalma, haber, ulaşım, öneri, borsa ...) ne olumlu ne olumsuz:
    atlanır (yanlış etiketle modele çelişki öğretilmesin).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

from . import REPO_ROOT, SHARED
from .protocol import ProtocolError, canonical, validate_request

# --------------------------------------------------------------------------- kaynaklar (adres + sha256 sabit)

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
MASSIVE_MEMBER = "1.1/data/tr-TR.jsonl"
MASSIVE_TAR_NAME = "amazon-massive-dataset-1.1.tar.gz"
# xSID: değişmesin diye commit'e sabitli (main = 761d926, 2026-10-30'da okundu)
XSID_COMMIT = "761d926c3f2e9f6bea13192d2f39cec1d04d9777"
XSID_URL = "https://raw.githubusercontent.com/mainlp/xsid/" + XSID_COMMIT + "/data/xSID-0.6/{name}"
XSID_FILES = {
    "tr.valid.conll": "0ac8d0d818c65eefd332f17d01f51e272ab1fce630216b7aa529291e716b83fc",
    "tr.test.conll": "c600eb35a716f28d682fe77610dc30a8f9e63612df48ebb52ed3f9e5e0f2cf21",
}

MASSIVE_LICENSE = "CC-BY-4.0"
XSID_LICENSE = "CC-BY-SA-4.0"
MASSIVE_ATTRIBUTION = ("MASSIVE 1.1 (tr-TR) © Amazon.com, Inc., CC BY 4.0 — FitzGerald ve ark. 2022, "
                       "\"MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset\"; SLURP (Bastianelli ve ark. 2020) "
                       "üzerine kurulu. Orion'da değiştirildi: Türkçe satırlar süzüldü ve Orion makine diline çevrildi.")
XSID_ATTRIBUTION = ("xSID 0.6 (tr), CC BY-SA 4.0 — van der Goot ve ark. 2021, \"From Masked Language Modeling to Translation: "
                    "Non-English Auxiliary Tasks Improve Zero-shot Spoken Language Understanding\" (github.com/mainlp/xsid). "
                    "Yalnızca değerlendirme; türetilmiş değerlendirme dosyası da CC BY-SA 4.0'dır.")

DATA_SOURCES = [
    {"id": "massive-1.1-tr-TR", "url": MASSIVE_URL, "sha256": MASSIVE_SHA256, "license": MASSIVE_LICENSE,
     "license_verified": True, "use": "train (train bölümü) + değerlendirme massive_tr (dev+test, yalnızca rapor)",
     "attribution": MASSIVE_ATTRIBUTION},
    {"id": "xsid-0.6-tr", "url": XSID_URL.format(name="tr.{valid,test}.conll"), "sha256": XSID_FILES,
     "license": XSID_LICENSE, "license_verified": True, "use": "yalnızca değerlendirme xsid_tr (valid+test, yalnızca rapor)",
     "attribution": XSID_ATTRIBUTION},
]

TRAIN_FILE = "massive-train.jsonl"
EVAL_FILES = {"massive_tr": "massive_tr.jsonl", "xsid_tr": "xsid_tr.jsonl"}

# Eğitim akışındaki pay (train.py Stream): tüm örneklerin %5'i (Tur 19 sahip kararı; önce %12). Bu payın içinde:
# eylem komutları %25, sohbet (ask, gerçekten soru) %30, kapsam dışı olumsuz (ask) %45.
# Eylem havuzu küçük (~370 satır): 10000 adım × 48 örnekte her eylem satırı ~16 kez görülür (önce 6000 adımda ~23).
# Sohbet + kapsam dışı ≈ akışın %3,75'i ask olur (önce %9; dilbilgisi üreticisinde ask ağırlığı ~%4).
# %12'de kapsam dışı "ask" olumsuzları ("salı günkü doktor randevusunu iptal edebilir misin", "alarmları kaldır")
# eylem komutlarıyla aynı yüzey ipuçlarını taşıdığından eylem cümlelerinde ilk belirteç güvenini bölüyordu (Tur 19 raporu).
EXT_FRAC = 0.05
POOL_WEIGHTS = {"action": 0.25, "chat": 0.3, "neg": 0.45}


def default_ext_dir() -> Path:
    import os
    return Path(os.environ.get("ORION_EXT_DATA") or (REPO_ROOT / "core-model" / "work" / "ext"))


# --------------------------------------------------------------------------- indirme / doğrulama

class DataError(RuntimeError):
    """İndirme ya da sha256 doğrulaması başarısız: sessiz geri dönüş yok, iş durur."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(p: Path, want: str) -> Path:
    if not p.exists():
        raise DataError(f"dosya yok: {p}")
    got = sha256_file(p)
    if got != want:
        raise DataError(f"sha256 uyuşmadı: {p} ({got} != {want})")
    return p


def fetch(url: str, dest: Path, want: str) -> Path:
    """Önbellekte doğru dosya varsa onu kullanır; yoksa indirir ve doğrular (uymazsa DataError)."""
    if dest.exists() and sha256_file(dest) == want:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except OSError as e:
        raise DataError(f"indirilemedi: {url}: {e}") from e
    got = sha256_file(tmp)
    if got != want:
        tmp.unlink(missing_ok=True)
        raise DataError(f"sha256 uyuşmadı: {url} ({got} != {want})")
    tmp.replace(dest)
    return dest


def read_massive_tar(tar_path: Path, member: str = MASSIVE_MEMBER) -> list[dict]:
    with tarfile.open(tar_path, "r:gz") as tf:
        f = tf.extractfile(member)
        if f is None:
            raise DataError(f"{member} tarball içinde yok")
        text = io.TextIOWrapper(f, encoding="utf-8").read()
    return [json.loads(x) for x in text.splitlines() if x.strip()]


# --------------------------------------------------------------------------- metin yardımcıları

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜÂÎÛ", "cgiosuaiuCGIOSUAIU")
WAKE = re.compile(r"(?<!\S)(?:olly|alexa)[,:!.]?(?!\S)", re.IGNORECASE)


def tr_lower(s: str) -> str:
    return s.translate(_TR_LOWER).lower()


def fold(s: str) -> str:
    """Küçük harf + Türkçe karakter katlama (anahtar sözcük eşleştirme için)."""
    return tr_lower(s).translate(_ASCII)


def norm_key(s: str) -> str:
    """Sızıntı denetimi anahtarı: uyandırma sözcüğü atılır, küçük harf, aksansız, noktalama yok, tek boşluk."""
    s = WAKE.sub(" ", s)
    s = re.sub(r"[^\p{L}\p{N}]+", " ", fold(s))
    return " ".join(s.split())


def strip_wake(s: str) -> str:
    """İngilizce uyandırma sözcüğü ("olly", "alexa") Türkçe cümlelerde çevrilmeden kalmış: atılır."""
    return " ".join(WAKE.sub(" ", s).split())


# Kotlin RuleTranslator.stripPolite ile aynı kural (sohbete giden soru metni)
_POLITE_PRE = re.compile(r"^(?:hey\s+orion[,:]?\s+|orion[,:]\s+|lütfen\s+|lutfen\s+|acaba\s+)+", re.IGNORECASE)
_POLITE_POST = re.compile(r"(?:[,\s]+(?:lütfen|lutfen|teşekkürler|tesekkurler|teşekkür ederim|rica etsem|sağ ol|sag ol))+[.!?\s]*$"
                          r"|[.!?\s]+$", re.IGNORECASE)


def strip_polite(s: str) -> str:
    t = s.strip()
    for _ in range(2):
        t = _POLITE_PRE.sub("", t)
        t = _POLITE_POST.sub("", t)
    return t.strip()


_SLOT = re.compile(r"\[(\w+) : ([^\]]*)\]")


def parse_annot(annot: str) -> list[tuple[str | None, str]]:
    """'[date : yarın] beni [time : sabah beşte] uyandır' → [('date','yarın'), (None,'beni'), ('time','sabah beşte'), (None,'uyandır')]"""
    out: list[tuple[str | None, str]] = []
    pos = 0
    for m in _SLOT.finditer(annot):
        pre = annot[pos:m.start()].strip()
        if pre:
            out.append((None, pre))
        out.append((m.group(1), m.group(2).strip()))
        pos = m.end()
    rest = annot[pos:].strip()
    if rest:
        out.append((None, rest))
    return out


def quality_ok(judgments: list[dict], need_slots: bool = False) -> bool:
    """Kalite süzgeci (MASSIVE yerelleştirme değerlendirmeleri, genelde 3 yargıç; ÇOĞUNLUK = yarıdan fazlası):
    intent_score == 1 (niyet doğru), grammar_score >= 3 (0-4 ölçeği; dilbilgisi iyi), spelling_score >= 1 (0 = bozuk),
    language_identification 'target' içerir (Türkçe; 'target|english' = içinde özel ad olan Türkçe cümle).
    Yuvalardan argüman çıkarılan satırlarda ayrıca slots_score ∈ {1, 2} (yuvalar doğru / yuva yok)."""
    if not judgments:
        return False

    def maj(f) -> bool:
        return sum(1 for j in judgments if f(j)) * 2 > len(judgments)

    ok = (maj(lambda j: j.get("intent_score") == 1) and maj(lambda j: (j.get("grammar_score") or 0) >= 3)
          and maj(lambda j: (j.get("spelling_score") or 0) >= 1)
          and maj(lambda j: "target" in str(j.get("language_identification", ""))))
    if ok and need_slots:
        ok = maj(lambda j: j.get("slots_score") in (1, 2))
    return ok


# --------------------------------------------------------------------------- zaman (Orion 'when')

_UNITS = {"bir": 1, "iki": 2, "uc": 3, "dort": 4, "dord": 4, "bes": 5, "alti": 6, "yedi": 7, "sekiz": 8, "dokuz": 9}
_TENS = {"on": 10, "yirmi": 20, "otuz": 30, "kirk": 40, "elli": 50}
_CASE = ("daki", "deki", "taki", "teki", "dan", "den", "tan", "ten", "da", "de", "ta", "te", "ya", "ye", "a", "e")
_DOW = {"pazartesi": "mon", "sali": "tue", "carsamba": "wed", "persembe": "thu", "cuma": "fri", "cumartesi": "sat",
        "pazar": "sun"}
_MONTHS = ["ocak", "subat", "mart", "nisan", "mayis", "haziran", "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik"]
_PARTS = {"sabah": "sabah", "sabahi": "sabah", "sabaha": "sabah", "ogle": "ogle", "oglen": "ogle", "oglene": "ogle",
          "ogleye": "ogle", "aksam": "aksam", "aksami": "aksam", "aksama": "aksam", "gece": "gece"}


def _num_token(t: str) -> tuple[int, str] | None:
    """'beşte' → (5,'te'), 'on' → (10,''), 'dörde' → (4,'e'), '17' → (17,''), '9'da' → (9,'da')."""
    t = t.replace("'", "")
    if t.isdigit():
        return int(t), ""
    m = re.fullmatch(r"(\d+)(" + "|".join(_CASE) + ")", t)
    if m:
        return int(m.group(1)), m.group(2)
    for w, v in list(_UNITS.items()) + list(_TENS.items()):
        if t == w:
            return v, ""
        if t.startswith(w) and t[len(w):] in _CASE:
            return v, t[len(w):]
        # yumuşama: dört → dörde, dörtte
    return None


def _words_number(toks: list[str], i: int, max_val: int) -> tuple[int, int, str] | None:
    """Ardışık sayı sözcükleri (on + birim) → (değer, tüketilen, ek). Ek yalnızca son sözcükte olabilir."""
    first = _num_token(toks[i]) if i < len(toks) else None
    if not first:
        return None
    v, suf = first
    n = 1
    if not suf and v in _TENS.values() and i + 1 < len(toks):
        nxt = _num_token(toks[i + 1])
        if nxt and nxt[0] < 10 and v + nxt[0] <= max_val:
            v, suf, n = v + nxt[0], nxt[1], 2
    if v > max_val:
        return None
    return v, n, suf


class WhenParse:
    def __init__(self):
        self.day: str | None = None      # d+N | dow | MM-DD
        self.part: str | None = None
        self.hour: int | None = None
        self.minute: int = 0
        self.offset: str | None = None
        self.repeat: str | None = None
        self.saat = False


def _parse_tokens(toks: list[str]) -> WhenParse | None:
    w = WhenParse()
    i = 0

    def set_day(v: str) -> bool:
        if w.day is not None and w.day != v:
            return False
        w.day = v
        return True

    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in ("bugun", "bugunku", "bugune"):
            if not set_day("d+0"):
                return None
            i += 1
        elif t == "bu" and (nxt in _PARTS or nxt in _DOW or nxt == "gun"):
            if nxt == "gun":
                if not set_day("d+0"):
                    return None
                i += 2
            else:
                if nxt in _PARTS and not set_day("d+0"):
                    return None
                i += 1
        elif t in ("yarin", "yarinki", "yarina"):
            if not set_day("d+1"):
                return None
            i += 1
        elif (t, nxt) in (("obur", "gun"), ("ertesi", "gun")) or (t, nxt) == ("yarindan", "sonra"):
            if not set_day("d+2"):
                return None
            i += 2
        elif t in ("gelecek", "onumuzdeki") and nxt in _DOW:
            i += 1
        elif t == "her" and nxt in ("gun", "sabah", "aksam", "gece"):
            w.repeat = "daily"
            if not set_day("d+0"):
                return None
            i += 2 if nxt == "gun" else 1
        elif t == "her" and nxt in _DOW:
            w.repeat = "weekly"
            i += 1
        elif t in _DOW:
            if not set_day(_DOW[t]):
                return None
            i += 1
            if i < len(toks) and toks[i] in ("gunu", "gunleri"):
                if toks[i] == "gunleri":
                    w.repeat = "weekly"
                i += 1
        elif t in _PARTS or (t, nxt) in (("ogleden", "sonra"), ("ogleden", "once")):
            if w.part is not None:
                return None
            if t == "ogleden":
                w.part = "ogleden " + nxt
                i += 2
            else:
                w.part = _PARTS[t]
                i += 1
        elif t == "saat":
            w.saat = True
            i += 1
        elif t == "yarim" and nxt == "saat" and i + 2 < len(toks) and toks[i + 2] == "sonra":
            w.offset = "+30m"
            i += 3
        elif re.fullmatch(r"\d{1,2}[:.]\d{2}(?:'?(?:da|de|ta|te|a|e|ya|ye))?", t):
            if w.hour is not None:
                return None
            m = re.match(r"(\d{1,2})[:.](\d{2})", t)
            w.hour, w.minute = int(m.group(1)), int(m.group(2))
            if w.hour > 23 or w.minute > 59:
                return None
            i += 1
        else:
            num = _words_number(toks, i, 31)
            if not num:
                return None
            v, n, suf = num
            j = i + n
            after = toks[j] if j < len(toks) else ""
            if not suf and after in ("dakika", "dk", "saat") and j + 1 < len(toks) and toks[j + 1] == "sonra":
                if w.offset or v < 1:
                    return None
                w.offset = f"+{v}{'m' if after in ('dakika', 'dk') else 'h'}"
                i = j + 2
                continue
            if not suf and after == "gun" and j + 1 < len(toks) and toks[j + 1] == "sonra":
                if not 1 <= v <= 30 or not set_day(f"d+{v}"):
                    return None
                i = j + 2
                continue
            mon = re.fullmatch(r"(" + "|".join(_MONTHS) + r")(?:da|de|ta|te|in|nin|nda|nde)?", after) if not suf else None
            if mon and 1 <= v <= 31:
                if not set_day(f"{_MONTHS.index(mon.group(1)) + 1:02d}-{v:02d}"):
                    return None
                i = j + 1
                continue
            # saat: saat sözcüğü + (dakika sözcükleri) + ek
            if w.hour is not None or v > 23:
                return None
            w.hour = v
            i = j
            if not suf and i < len(toks):
                b = toks[i]
                if re.fullmatch(r"bucu(?:k|kta|ga)", b):
                    w.minute = 30
                    i += 1
                    continue
                mins = _words_number(toks, i, 59)
                if mins and mins[0] >= 10 and (mins[2] or i + mins[1] == len(toks)):
                    w.minute = mins[0]
                    i += mins[1]
    return w


def parse_when(phrase: str, default_hour: int | None = None) -> tuple[str, str | None] | None:
    """Türkçe zaman ifadesi → (when, repeat). Yalnızca TAMAMI tanınan ifadeler; belirsiz saat (gün bölümü yok, 1-6)
    ve gün bölümüyle çelişen saat → None. Saat yoksa ve default_hour verilmişse gün@default (Kotlin TimeParser:
    hatırlatıcıda "yarın dişçi" → 09:00; takvimde saat zorunlu)."""
    toks = [t for t in re.split(r"[\s,]+", fold(phrase).replace("’", "'")) if t]
    if not toks:
        return None
    w = _parse_tokens(toks)
    if w is None:
        return None
    if w.offset:
        if w.day or w.part or w.hour is not None or w.repeat:
            return None
        return w.offset, None
    h = w.hour
    if h is not None:
        p = w.part
        if p == "sabah":
            if not 1 <= h <= 11:
                return None
        elif p == "ogleden once":
            if not 1 <= h <= 12:
                return None
        elif p == "ogle":
            if h == 12:
                pass
            elif 1 <= h <= 5:
                h += 12
            elif not 12 <= h <= 17:
                return None
        elif p == "ogleden sonra":
            if 1 <= h <= 11:
                h += 12
            elif not 12 <= h <= 18:
                return None
        elif p == "aksam":
            if 1 <= h <= 11:
                h += 12
            elif not 17 <= h <= 23:
                return None
        elif p == "gece":
            if 6 <= h <= 11:
                h += 12
            elif not 18 <= h <= 23:
                return None
        else:
            if 1 <= h <= 6 or h == 0:
                return None  # sabah mı akşam mı belli değil
        clock = f"{h:02d}:{w.minute:02d}"
    elif w.part == "ogle":
        clock = "12:00"
    elif w.part is not None:
        return None  # "yarın akşam" saatsiz: varsayılan 09:00 akşamla çelişir
    elif w.day is not None and default_hour is not None:
        clock = f"{default_hour:02d}:00"
    else:
        return None
    day = w.day or "d+0"
    return f"{day}@{clock}", w.repeat


def parse_day(phrase: str) -> str | None:
    """Takvim listesi günü (protokol 'date': d+N). Yalnızca bugün/yarın/öbür gün/N gün sonra."""
    f = " ".join(fold(phrase).split())
    table = {"bugun": "d+0", "bu gun": "d+0", "bugunku": "d+0", "yarin": "d+1", "yarinki": "d+1", "obur gun": "d+2",
             "ertesi gun": "d+2", "yarindan sonra": "d+2"}
    if f in table:
        return table[f]
    m = re.fullmatch(r"(\S+(?: \S+)?) gun sonra", f)
    if m:
        num = _words_number(m.group(1).split(), 0, 30)
        if num and num[1] == len(m.group(1).split()) and not num[2]:
            return f"d+{num[0]}"
    return None


# --------------------------------------------------------------------------- niyet → komut

class Skip(Exception):
    pass


# Orion'da komutu olmayan niyetler → kapsam dışı (ask)
NEG_INTENTS = {
    "alarm_remove", "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up",
    "calendar_remove", "datetime_convert", "datetime_query", "email_addcontact", "email_querycontact",
    "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton",
    "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on", "lists_remove", "music_query", "music_settings", "play_game",
    "social_post", "social_query", "takeaway_order", "takeaway_query", "transport_taxi", "transport_ticket",
    "weather_query",
}
# Gerçekten soru/sohbet → ask doğru komut (olumlu)
CHAT_INTENTS = {"qa_factoid", "qa_definition", "qa_currency", "cooking_recipe", "cooking_query", "general_joke",
                "general_greet", "general_quirky", "recommendation_movies"}
# Yaklaşık karşılık: ne olumlu ne olumsuz (atlanır)
SKIP_INTENTS = {"play_music", "play_radio", "play_podcasts", "play_audiobook", "news_query", "recommendation_events",
                "recommendation_locations", "transport_query", "transport_traffic", "qa_stock", "music_likeness",
                "music_dislikeness", "email_sendemail"}
ACTION_INTENTS = {"alarm_set", "alarm_query", "calendar_set", "calendar_query", "email_query", "lists_createoradd",
                  "lists_query", "qa_maths"}

# Yuva dışında kalabilecek sözcükler (katlanmış). Başka sözcük kalırsa argüman eksik demektir → atla.
_G_COMMON = {"bana", "beni", "benim", "icin", "lutfen", "bir", "de", "da", "ve", "misin", "musun", "miyim", "mi", "mu",
             "ben", "sen", "senden", "istiyorum", "isterim", "rica", "ederim", "olarak", "saat", "lazim", "gerek",
             "gerekiyor", "sonra", "diye", "ile", "ki", "hey", "simdi", "su", "bu", "gelecek", "onumuzdeki"}
_G_REM = _G_COMMON | {
    "hatirlat", "hatirlatir", "hatirlatin", "hatirlatabilir", "hatirlatici", "hatirlaticisi", "hatirlaticiyi",
    "hatirlatma", "hatirlatmasi", "hatirlatmayi", "ayarla", "ayarlayin", "ayarlar", "ayarlayabilir", "kur", "kurar",
    "kurun", "kurabilir", "olustur", "olusturur", "alarm", "alarmi", "alarmini", "alarmim", "alarmimi", "uyar",
    "uyandir", "haber", "ver", "unutma", "unutmamami", "unutmayayim", "olsun", "yap", "ekle", "gun", "gunu", "zamani",
    "olacak", "var", "bildir", "oldugunu", "animsat", "hatirlatabilir"}
_G_CAL = _G_COMMON | {
    "takvime", "takvimime", "takvimimde", "takvimine", "takvimimize", "takvim", "ajandama", "ajandaya", "ekle",
    "ekler", "ekleyin", "ekleyebilir", "koy", "yaz", "kaydet", "planla", "programla", "programima", "olustur",
    "ayarla", "etkinlik", "etkinligi", "etkinligini", "isaretle", "isaretler", "var", "olacak", "yeni", "gun", "gunu"}


def _residual(segs: list[tuple[str | None, str]], used: set[str], glue: set[str], after_ok=("sonra",)) -> list[str]:
    """Kullanılan yuvalar dışındaki, bağlaç kümesinde olmayan sözcükler."""
    bad = []
    for slot, text in segs:
        if slot is not None and slot in used:
            continue
        if slot is not None:
            bad.append(f"[{slot}]")
            continue
        for wd in re.split(r"[\s,.!?]+", fold(text)):
            if wd and wd not in glue and wd not in after_ok:
                bad.append(wd)
    return bad


_TIME_SLOTS = ("date", "time", "timeofday", "general_frequency")


def _time_phrase(segs: list[tuple[str | None, str]]) -> str:
    parts = []
    for i, (slot, text) in enumerate(segs):
        if slot in _TIME_SLOTS:
            parts.append(text)
            # "[time : üç saat] sonra" → yuva dışındaki 'sonra' göreli zamana aittir
            if i + 1 < len(segs) and segs[i + 1][0] is None:
                nxt = fold(segs[i + 1][1]).split()
                if nxt and nxt[0] == "sonra":
                    parts.append("sonra")
                if nxt and nxt[0] in ("once", "kala", "gece", "gecerken"):
                    raise Skip("relative_to_event")
                # "[time : öğlen] bir toplantı" → öğlen bir (13:00) mi, bir toplantı mı? belirsiz
                if nxt and _num_token(nxt[0]) and not re.fullmatch(r"\d+", nxt[0]):
                    raise Skip("time_ambiguous")
    return " ".join(parts)


_REM_CUE = re.compile(r"hatirlat|unutma|haber ver|\buyar\b|uyandir|\balarm")
_INF = (("mayi", "mak"), ("meyi", "mek"), ("mami", "mak"), ("memi", "mek"))


def _text_slot(segs, names) -> str:
    vals = [t for s, t in segs if s in names]
    if len(vals) != 1:
        raise Skip("no_text" if not vals else "multi_text")
    v = vals[0].strip(" ,.'")
    words = v.split()
    if words:
        last = words[-1]
        fl = fold(last)
        for a, b in _INF:
            if fl.endswith(a) and len(fl) > len(a) + 1:  # Kotlin RuleTranslator.infinitive ile aynı
                words[-1] = last[: -len(a)] + b
                break
    v = " ".join(words)
    if not v:
        raise Skip("no_text")
    return v


def _reminder(utt: str, segs, text_slots=("event_name",)) -> tuple[str, dict]:
    phrase = _time_phrase(segs)
    if not phrase:
        raise Skip("no_time")
    try:
        text = _text_slot(segs, text_slots)
    except Skip:
        # Metin yok ama "10 dakika sonra" → geri sayım (timer.start yalnızca dakika ister; uydurma metin yok)
        pw = parse_when(phrase)
        if pw and pw[0].startswith("+") and pw[0][-1] in "mh":
            n = int(pw[0][1:-1]) * (60 if pw[0][-1] == "h" else 1)
            bad = _residual(segs, set(_TIME_SLOTS), _G_REM)
            if bad:
                raise Skip("residual_words")
            if 1 <= n <= 1440:
                return "timer.start", {"minutes": n}
        raise
    pw = parse_when(phrase, default_hour=9)
    if pw is None:
        raise Skip("time_unparsed")
    bad = _residual(segs, set(_TIME_SLOTS) | set(text_slots), _G_REM)
    if bad:
        raise Skip("residual_words")
    args = {"at": pw[0], "text": text}
    if pw[1]:
        args["repeat"] = pw[1]
    return "reminder.set", args


def _calendar_add(utt: str, segs) -> tuple[str, dict]:
    phrase = _time_phrase(segs)
    if not phrase:
        raise Skip("no_time")
    title = _text_slot(segs, ("event_name",))
    pw = parse_when(phrase, default_hour=None)
    if pw is None or pw[0].startswith("+"):
        raise Skip("time_unparsed")
    if pw[1]:
        raise Skip("repeat_unsupported")  # calendar.add tekrar argümanı yok
    bad = _residual(segs, set(_TIME_SLOTS) | {"event_name"}, _G_CAL)
    if bad:
        raise Skip("residual_words")
    return "calendar.add", {"title": title, "start": pw[0]}


_CAL_CUE = re.compile(r"takvim|ajanda|program|etkinlik|toplanti|randevu|\bplan|ne var|isim var|mesgul|bos muyum")
_MAIL_CUE = re.compile(r"e-?posta|\be?mail|\bposta|ileti")


def _calendar_list(utt: str, segs) -> tuple[str, dict]:
    slots = [(s, t) for s, t in segs if s]
    if len(slots) != 1 or slots[0][0] != "date":
        raise Skip("slots_unsupported")
    if not _CAL_CUE.search(fold(utt)):
        raise Skip("no_cue")
    d = parse_day(slots[0][1])
    if d is None:
        raise Skip("time_unparsed")
    return "calendar.list", {"day": d}


def _mail_check(utt: str, segs) -> tuple[str, dict]:
    if not _MAIL_CUE.search(fold(utt)):
        raise Skip("no_cue")
    # birleşik ya da başka iş içeren istek (konu süzgeci, yıldızla, sil, cevapla, ilet ...) → mail.check eksik kalır
    if re.search(r"\bkonu|\bve\b|yildiz|\bsil|gonder|cevap|yanit|ilet|\bat\b|isaretle", fold(utt)):
        raise Skip("residual_words")
    slots = [(s, t) for s, t in segs if s]
    if not slots:
        return "mail.check", {}
    if len(slots) == 1 and slots[0][0] == "person" and "'" in slots[0][1]:
        name = slots[0][1].split("'")[0].strip()
        if 2 <= len(name) <= 80 and fold(slots[0][1].split("'")[1]) in ("dan", "den", "tan", "ten"):
            return "mail.check", {"query": name}
    raise Skip("slots_unsupported")


_LIST_ADD = re.compile(r"^(?P<item>.+?) (?:ekle|ekler misin|ekleyin|ekleyebilir misin|koy)$")


def _note_write(utt: str, segs) -> tuple[str, dict]:
    """'[list_name : alışveriş] listeme muz ekle' → note.write {text: muz, tags: [alışveriş]} (Orion'da liste = etiketli not)."""
    if len(segs) != 2 or segs[0][0] != "list_name" or segs[1][0] is not None:
        raise Skip("pattern_unsupported")
    name, rest = segs[0][1], segs[1][1]
    m = re.match(r"^(?:listeme|listesine|listemize) (.+)$", rest)
    if not m:
        raise Skip("pattern_unsupported")
    m2 = _LIST_ADD.match(m.group(1))
    if not m2 or len(m2.group("item").split()) > 4 or len(name) > 32:
        raise Skip("pattern_unsupported")
    item = m2.group("item")
    if re.search(r"(?:mem|mam)\b|gerek|lazim|lutfen|\boge|^bir |^yeni |^bunu|^sunu", fold(item)) or fold(name) == "alarm":
        raise Skip("pattern_unsupported")
    return "note.write", {"text": item, "tags": [name]}


def _note_search(utt: str, segs) -> tuple[str, dict]:
    """'[list_name : market] listemi göster/oku' → note.search {query: market}."""
    if len(segs) != 2 or segs[0][0] != "list_name" or segs[1][0] is not None:
        raise Skip("pattern_unsupported")
    if not re.fullmatch(r"(?:listemi|listesini) (?:goster|oku|getir|ac|kontrol et|soyle)", fold(segs[1][1])):
        raise Skip("pattern_unsupported")
    return "note.search", {"query": segs[0][1]}


_OPS = {"arti": "+", "eksi": "-", "carpi": "*", "kere": "*", "bolu": "/"}
_BIG = {"yuz": 100, "bin": 1000}


def words_to_int(words: list[str]) -> int | None:
    """'yüz yirmi üç' → 123, 'iki yüz' → 200, 'bin beş yüz' → 1500. Yalnızca sayı sözcükleri/rakamlar."""
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    total, cur = 0, 0
    if not words:
        return None
    for w in words:
        if w in _UNITS and w != "dord":
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "yuz":
            cur = (cur or 1) * 100
        elif w == "bin":
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
    return total + cur


def _calc(utt: str, segs) -> tuple[str, dict]:
    f = " ".join(re.split(r"[\s?]+", fold(strip_polite(utt)))).strip()
    m = re.fullmatch(r"(?P<a>[a-z0-9 ]+?) (?P<op>arti|eksi|carpi|kere|bolu) (?P<b>[a-z0-9 ]+?) "
                     r"(?:nedir|kac|kactir|kac eder|ne eder|ne yapar|kac yapar|esittir ne)", f)
    if not m:
        raise Skip("pattern_unsupported")
    a, b = words_to_int(m.group("a").split()), words_to_int(m.group("b").split())
    if a is None or b is None:
        raise Skip("pattern_unsupported")
    return "calc.eval", {"expr": f"{a}{_OPS[m.group('op')]}{b}"}


def _reminder_list(utt: str, segs) -> tuple[str, dict]:
    if any(s for s, _ in segs):
        raise Skip("slots_unsupported")  # reminder.list'in gün/tür argümanı yok
    if not re.search(r"alarm|hatirlatic|hatirlatma", fold(utt)):
        raise Skip("no_cue")
    if re.search(r"\b(sil|iptal|kapat|kaldir|degistir|kur|ayarla|ertele)\b", fold(utt)):
        raise Skip("no_cue")
    return "reminder.list", {}


def convert_massive(intent: str, utt: str, segs) -> tuple[str, dict, str]:
    """→ (cmd, args, kind) kind ∈ action|chat|neg. Skip(reason) fırlatabilir."""
    if intent in NEG_INTENTS:
        q = strip_polite(utt)
        return "ask", {"question": q}, "neg"
    if intent in CHAT_INTENTS:
        return "ask", {"question": strip_polite(utt)}, "chat"
    if intent in SKIP_INTENTS:
        raise Skip("approximate_intent")
    if intent == "alarm_set":
        cmd, args = _reminder(utt, segs)
    elif intent == "calendar_set":
        cmd, args = _reminder(utt, segs) if _REM_CUE.search(fold(utt)) else _calendar_add(utt, segs)
    elif intent == "alarm_query":
        cmd, args = _reminder_list(utt, segs)
    elif intent == "calendar_query":
        cmd, args = _calendar_list(utt, segs)
    elif intent == "email_query":
        cmd, args = _mail_check(utt, segs)
    elif intent == "lists_createoradd":
        cmd, args = _note_write(utt, segs)
    elif intent == "lists_query":
        cmd, args = _note_search(utt, segs)
    elif intent == "qa_maths":
        cmd, args = _calc(utt, segs)
    else:
        raise Skip("unknown_intent")
    return cmd, args, "action"


def make_row(src: str, cmd: str, args: dict, **meta) -> dict:
    """Doğrulayıcıdan geçmeyen satır üretilmez (ProtocolError → Skip)."""
    try:
        full = canonical(validate_request({"v": 1, "id": "g", "cmd": cmd, "args": args}))
    except ProtocolError as e:
        raise Skip(f"invalid:{e.args[0] if e.args else 'protocol'}") from e
    if cmd == "ask" and len(args.get("question", "")) < 2:
        raise Skip("invalid:short")
    return {"task": "t2c", "src": src, "tgt": canonical({"cmd": cmd, "args": args}), "full": full, "cmd": cmd, **meta}


def massive_rows(raw: list[dict]) -> tuple[list[dict], Counter, Counter]:
    """Tüm tr-TR satırları → çevrilmiş satırlar (bölüm bilgisiyle). Döner: (rows, skipped_by_reason, kept_by_cmd)."""
    out, skipped, kept = [], Counter(), Counter()
    for r in raw:
        if r.get("locale") != "tr-TR":
            continue
        intent = r["intent"]
        need_slots = intent in ACTION_INTENTS
        if not quality_ok(r.get("judgments", []), need_slots=need_slots):
            skipped["quality"] += 1
            continue
        segs = [(s, strip_wake(t)) for s, t in parse_annot(r["annot_utt"])]
        segs = [(s, t) for s, t in segs if t]
        utt = strip_wake(r["utt"])
        if " ".join(t for _, t in segs).split() != utt.split():
            skipped["annot_mismatch"] += 1
            continue
        if len(utt) < 2:
            skipped["empty"] += 1
            continue
        try:
            cmd, args, kind = convert_massive(intent, utt, segs)
            row = make_row(utt, cmd, args, kind=kind, intent=intent, partition=r["partition"],
                           source=f"massive-1.1/tr-TR#{r['id']}", license=MASSIVE_LICENSE)
        except Skip as e:
            skipped[str(e)] += 1
            continue
        kept[f"{cmd}/{kind}"] += 1
        out.append(row)
    return out, skipped, kept


# --------------------------------------------------------------------------- xSID

XSID_NEG = {"weather/find", "alarm/cancel_alarm", "alarm/modify_alarm", "alarm/snooze_alarm", "alarm/time_left_on_alarm",
            "reminder/cancel_reminder", "AddToPlaylist", "BookRestaurant", "RateBook", "SearchCreativeWork",
            "SearchScreeningEvent"}
XSID_SKIP = {"PlayMusic"}  # MASSIVE play_music ile aynı karar


def parse_conll(text: str) -> list[dict]:
    """xSID conll: '# text = ...', '# intent = ...', satırlar: no<TAB>sözcük<TAB>niyet<TAB>BIO-yuva."""
    out, cur = [], None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if cur and cur.get("tokens"):
                out.append(cur)
            cur = None
            continue
        if cur is None:
            cur = {"tokens": [], "tags": []}
        if line.startswith("# "):
            k, _, v = line[2:].partition(" = ")
            cur[k.strip()] = v.strip()
            continue
        cols = line.split("\t")
        if len(cols) >= 4:
            cur["tokens"].append(cols[1])
            cur["tags"].append(cols[3])
    return out


def xsid_segments(rec: dict) -> list[tuple[str | None, str]]:
    """BIO yuvaları → MASSIVE biçimli parçalar (yuva adı: datetime → time, reminder/todo → event_name)."""
    rename = {"datetime": "time", "reminder/todo": "event_name"}
    segs: list[tuple[str | None, list[str]]] = []
    for tok, tag in zip(rec["tokens"], rec["tags"]):
        if tag.startswith("B-"):
            segs.append((rename.get(tag[2:], tag[2:]), [tok]))
        elif tag.startswith("I-") and segs and segs[-1][0] == rename.get(tag[2:], tag[2:]):
            segs[-1][1].append(tok)
        elif segs and segs[-1][0] is None:
            segs[-1][1].append(tok)
        else:
            segs.append((None, [tok]))
    return [(s, " ".join(w for w in ws if not re.fullmatch(r"[?!.,]", w)).strip()) for s, ws in segs
            if any(not re.fullmatch(r"[?!.,]", w) for w in ws)]


def xsid_rows(text: str, split: str) -> tuple[list[dict], Counter, Counter]:
    out, skipped, kept = [], Counter(), Counter()
    for rec in parse_conll(text):
        intent, src = rec.get("intent", ""), rec.get("text", "").strip()
        segs = xsid_segments(rec)
        try:
            if intent in XSID_NEG:
                cmd, args, kind = "ask", {"question": strip_polite(src)}, "neg"
            elif intent in XSID_SKIP:
                raise Skip("approximate_intent")
            elif intent in ("alarm/set_alarm", "reminder/set_reminder"):
                cmd, args = _reminder(src, segs)
                kind = "action"
            elif intent in ("alarm/show_alarms", "reminder/show_reminders"):
                cmd, args = _reminder_list(src, segs)
                kind = "action"
            else:
                raise Skip("unknown_intent")
            # Kopyalanan serbest metin cümlede birebir geçmeli (BIO sözcükleri noktalamayla ayrılmış olabilir)
            for k in ("text", "question"):
                v = args.get(k)
                if v is not None and v not in src and not (v.endswith(("mek", "mak")) and v[:-1] in src):
                    raise Skip("span_not_in_text")
            row = make_row(src, cmd, args, kind=kind, intent=intent, partition=split,
                           source=f"xsid-0.6/tr.{split}#{rec.get('id', '?')}", license=XSID_LICENSE)
        except Skip as e:
            skipped[str(e)] += 1
            continue
        kept[f"{cmd}/{kind}"] += 1
        out.append(row)
    return out, skipped, kept


# --------------------------------------------------------------------------- sızıntı

def held_out_keys() -> set[str]:
    """Depodaki ölçüm kümeleri (elle yazılmış, kör) — dış eğitim verisi bunlarla çakışamaz."""
    keys = set()
    for name in ("handwritten-eval.jsonl", "blind-eval.jsonl"):
        p = SHARED / "core-model" / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    keys.add(norm_key(json.loads(line)["src"]))
    return keys


# Tur 18 denetiminde ayrılmış (OOD) kalıplara yakın bulunup eğitimden çıkarılan ifadeler: gerçek cümlede de geçerse
# o satır eğitime girmez (OOD ölçümü şişmesin). tests/test_datagen_tur18.py REMOVED_LITERALS ile aynı liste.
HELD_OUT_LITERALS = ["olacak, ", "unutmamalıyım", "atlamayayım", "ihmal etmeyeyim", "aklımdan çıkmasın",
                     "güncellemen var mı", "gelişim için öneri", "gelişimin için öneri", "fotoğrafa göz at", "planıma ekle",
                     "planımda ne var", "içerenleri", "konulu olanları", "dk'lık", "hatırlatma oluştur", "alarm oluştur",
                     "bildirim oluştur", "unutmamak için not al", "unutmayayım diye"]


def split_train_eval(rows: list[dict], xsid: list[dict], extra_keys: set[str] | None = None) -> tuple[list[dict], list[dict], Counter]:
    """MASSIVE train → eğitim; dev+test → massive_tr. Eğitimden çıkarılanlar: değerlendirme kümelerinden herhangi
    birinde (MASSIVE dev/test, xSID, elle yazılmış/kör küme) normalize hâli geçen cümleler, ayrılmış kalıp ifadeleri,
    tekrarlar."""
    ev = [r for r in rows if r["partition"] in ("dev", "test")]
    eval_keys = {norm_key(r["src"]) for r in ev} | {norm_key(r["src"]) for r in xsid} | (extra_keys or set())
    train, seen, dropped = [], set(), Counter()
    for r in rows:
        if r["partition"] != "train":
            continue
        k = norm_key(r["src"])
        if k in eval_keys:
            dropped["leak_eval"] += 1
            continue
        if any(lit in r["src"] for lit in HELD_OUT_LITERALS):
            dropped["held_out_literal"] += 1
            continue
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        train.append({**r, "training_allowed": True, "origin": "dataset"})
    ev_out, seen_ev = [], set()
    for r in ev:
        k = norm_key(r["src"])
        if k in seen_ev:
            continue
        seen_ev.add(k)
        ev_out.append({**r, "training_allowed": False, "origin": "dataset"})
    return train, ev_out, dropped


# --------------------------------------------------------------------------- hazırla / yükle

def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def prepare(out: Path, massive_tar: Path | None = None, xsid_dir: Path | None = None, download: bool = False,
            massive_sha: str = MASSIVE_SHA256, xsid_sha: dict | None = None) -> dict:
    """Kaynakları doğrular (indirir) ve eğitim/değerlendirme dosyalarını yazar. Her hata DataError: iş durur."""
    xsid_sha = xsid_sha or XSID_FILES
    raw_dir = out / "raw"
    if massive_tar is None:
        massive_tar = raw_dir / MASSIVE_TAR_NAME
        if download:
            fetch(MASSIVE_URL, massive_tar, massive_sha)
    verify(massive_tar, massive_sha)
    xs_text = {}
    for name, sha in xsid_sha.items():
        p = (xsid_dir / name) if xsid_dir else raw_dir / name
        if xsid_dir is None and download:
            fetch(XSID_URL.format(name=name), p, sha)
        xs_text[name] = verify(p, sha).read_text(encoding="utf-8")
    rows, skipped, kept = massive_rows(read_massive_tar(massive_tar))
    xs, xs_skipped, xs_kept = [], Counter(), Counter()
    for name, text in xs_text.items():
        r, s, k = xsid_rows(text, name.split(".")[1])
        xs += r
        xs_skipped.update(s)
        xs_kept.update(k)
    seen, xs_u = set(), []
    for r in xs:
        if norm_key(r["src"]) not in seen:
            seen.add(norm_key(r["src"]))
            xs_u.append(r)
    xs_u = [{**r, "training_allowed": False, "origin": "dataset"} for r in xs_u]
    train, ev, dropped = split_train_eval(rows, xs_u, held_out_keys())
    _write(out / TRAIN_FILE, train)
    _write(out / EVAL_FILES["massive_tr"], ev)
    _write(out / EVAL_FILES["xsid_tr"], xs_u)
    stats = {
        "massive": {"kept_by_cmd_kind_all_partitions": dict(sorted(kept.items())), "skipped": dict(skipped.most_common()),
                    "train_rows": len(train), "train_by_cmd_kind": dict(sorted(Counter(f"{r['cmd']}/{r['kind']}" for r in train).items())),
                    "train_dropped": dict(dropped), "eval_rows": len(ev),
                    "eval_by_cmd_kind": dict(sorted(Counter(f"{r['cmd']}/{r['kind']}" for r in ev).items()))},
        "xsid": {"kept_by_cmd_kind": dict(sorted(xs_kept.items())), "skipped": dict(xs_skipped.most_common()),
                 "eval_rows": len(xs_u)},
        "sources": DATA_SOURCES,
        "ext_frac": EXT_FRAC, "pool_weights": POOL_WEIGHTS,
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def load_train(ext_dir: Path) -> list[dict]:
    """Eğitim satırları; değerlendirme dosyaları da olmalı ve hiçbir eğitim cümlesi onlarda geçmemeli (yoksa DataError)."""
    p = ext_dir / TRAIN_FILE
    if not p.exists():
        raise DataError(f"{p} yok: önce 'python -m orion_core.massive prepare --out {ext_dir} --download'")
    rows = [r for r in _read(p) if r.get("training_allowed") is True]
    keys = held_out_keys()
    for name in EVAL_FILES.values():
        q = ext_dir / name
        if not q.exists():
            raise DataError(f"{q} yok: sızıntı denetimi yapılamaz")
        keys |= {norm_key(r["src"]) for r in _read(q)}
    leak = [r["src"] for r in rows if norm_key(r["src"]) in keys]
    if leak:
        raise DataError(f"eğitim/değerlendirme çakışması: {leak[:5]}")
    return [{"task": "t2c", "src": r["src"], "tgt": r["tgt"], "pool": r["kind"]} for r in rows]


def load_eval_sets(ext_dir: Path | None = None) -> dict[str, list[dict]]:
    """Hazırlanmış değerlendirme kümeleri (yoksa boş: değerlendirme bunları atlar ve 'atlandı' yazar)."""
    d = ext_dir or default_ext_dir()
    return {k: _read(d / v) for k, v in EVAL_FILES.items() if (d / v).exists()}


class ExtSampler:
    """Eğitim akışı için havuzlu örnekleyici (eylem / sohbet / kapsam dışı)."""

    def __init__(self, rows: list[dict], weights: dict | None = None):
        self.pools: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            self.pools[r.get("pool", "action")].append({"task": "t2c", "src": r["src"], "tgt": r["tgt"]})
        w = weights or POOL_WEIGHTS
        self.names = [k for k in w if self.pools.get(k)]
        self.weights = [w[k] for k in self.names]

    def __bool__(self) -> bool:
        return bool(self.names)

    def sample(self, rng: random.Random) -> dict:
        pool = rng.choices(self.names, weights=self.weights)[0]
        return rng.choice(self.pools[pool])


# --------------------------------------------------------------------------- değerlendirme (yalnızca rapor)

def score_ext(rows: list[dict], outs: list[str], normalize) -> dict:
    """Tam eşleşme, geçerlilik, komut/tür kırılımı ve yanlış-iş (beklenen dışında ask olmayan geçerli komut)."""
    per_cmd: dict = defaultdict(lambda: [0, 0])
    per_kind: dict = defaultdict(lambda: [0, 0])
    ok = valid = 0
    hits, wrongs, fails = [], [], []
    for r, o in zip(rows, outs):
        norm = normalize(o)
        hit = norm == r["full"]
        hits.append(hit)
        per_cmd[r["cmd"]][1] += 1
        per_kind[r["kind"]][1] += 1
        # yanlış iş: geçerli, beklenenden farklı ve sohbet (ask) olmayan bir komut çalışırdı
        wrongs.append(norm is not None and not hit and json.loads(norm)["cmd"] != "ask")
        if norm is not None:
            valid += 1
        if hit:
            ok += 1
            per_cmd[r["cmd"]][0] += 1
            per_kind[r["kind"]][0] += 1
        elif len(fails) < 15:
            fails.append({"src": r["src"], "want": r["tgt"], "got": o})
    n = max(1, len(rows))
    return {"exact": ok / n, "valid": valid / n, "wrong_actions": sum(wrongs) / n, "n": len(rows),
            "per_cmd": {k: round(v[0] / v[1], 3) for k, v in sorted(per_cmd.items())},
            "per_kind": {k: round(v[0] / v[1], 3) for k, v in sorted(per_kind.items())},
            "n_per_kind": {k: v[1] for k, v in sorted(per_kind.items())},
            "set_sha256": hashlib.sha256("".join(json.dumps([r["src"], r["full"]], ensure_ascii=False) + "\n"
                                                 for r in rows).encode()).hexdigest(),
            "fails": fails, "_hits": hits, "_wrong": wrongs}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MASSIVE/xSID → Orion eğitim ve değerlendirme dosyaları")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", default=str(default_ext_dir()))
    p.add_argument("--massive-tar", help="yerel amazon-massive-dataset-1.1.tar.gz (sha256 yine doğrulanır)")
    p.add_argument("--xsid-dir", help="tr.valid.conll ve tr.test.conll içeren yerel klasör (sha256 yine doğrulanır)")
    p.add_argument("--download", action="store_true", help="eksikse sabit adreslerden indir")
    a = ap.parse_args(argv)
    try:
        stats = prepare(Path(a.out), Path(a.massive_tar) if a.massive_tar else None,
                        Path(a.xsid_dir) if a.xsid_dir else None, a.download)
    except DataError as e:
        print(f"HATA: {e}", file=sys.stderr)
        sys.exit(2)
    s = {k: v for k, v in stats.items() if k != "sources"}
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
