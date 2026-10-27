"""Dilbilgisinden PROGRAMLA eğitim verisi üretir (kota ve lisans sorunu yok).

Üç görev:
  t2c  Türkçe istek -> komut (çevirici görevi)
  plan komut -> ajan + adım planı (araç seçimi)
  ans  cevap -> sabit biçimli Türkçe cevap

Ayrımlar:
  train      eğitim şablonları + eğitim değerleri
  test_iid   eğitim şablonları + ayrılmış değerler (yeni içerik)
  test_ood   ayrılmış şablonlar (hiç görülmemiş söyleyiş)
Her örnek kendi doğrulayıcımızdan geçer; geçmeyen üretilmez.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable

from .protocol import canonical, validate_request

# --------------------------------------------------------------------------- zaman

DOW_TR = [("pazartesi", "mon"), ("salı", "tue"), ("çarşamba", "wed"), ("perşembe", "thu"),
          ("cuma", "fri"), ("cumartesi", "sat"), ("pazar", "sun")]
MONTHS_TR = ["ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos",
             "eylül", "ekim", "kasım", "aralık"]
MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Sayının okunuşunun son hecesine göre bulunma eki ('de/'da/'te/'ta)
_UNIT_SUFFIX = {0: None, 1: "de", 2: "de", 3: "te", 4: "te", 5: "te", 6: "da", 7: "de", 8: "de", 9: "da"}
_TENS_SUFFIX = {1: "da", 2: "de", 3: "da", 4: "ta", 5: "de"}


def loc_suffix(n: int) -> str:
    """Saat/dakika okunuşuna göre bulunma eki: 3'te, 10'da, 14'te, 30'da, 45'te."""
    if n == 0:
        return "da"  # "sıfır" yerine "on iki"/"tam" gibi okunur; 00 için sıfırda
    u = n % 10
    if u:
        return _UNIT_SUFFIX[u]
    return _TENS_SUFFIX[(n // 10) % 10] if n < 60 else "de"


@dataclass
class TimeExpr:
    tr: str
    when: str


def _clock_phrase(rng: random.Random, h: int, m: int) -> str:
    """Saat ifadesi; eki ile birlikte. h 0-23."""
    style = rng.random()
    if m == 0:
        if style < 0.35 and 6 <= h <= 11:
            return f"sabah {h}'{loc_suffix(h)}"
        if style < 0.35 and 13 <= h <= 17:
            hh = h - 12
            return f"öğleden sonra {hh}'{loc_suffix(hh)}"
        if style < 0.35 and 18 <= h <= 22:
            hh = h - 12
            return f"akşam {hh}'{loc_suffix(hh)}"
        if style < 0.7:
            return f"saat {h}'{loc_suffix(h)}"
        return f"saat {h:02d}:00'{loc_suffix(h)}"
    if m == 30 and style < 0.3:
        return f"saat {h} buçukta"
    return f"saat {h:02d}:{m:02d}'{loc_suffix(m)}"


def random_time(rng: random.Random, allow_offset: bool = True) -> TimeExpr:
    h = rng.choice([7, 8, 9, 9, 10, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    m = rng.choice([0, 0, 0, 0, 15, 30, 30, 45, 10, 20, 40, 50, 5])
    hm = f"{h:02d}:{m:02d}"
    clock = _clock_phrase(rng, h, m)
    kind = rng.random()
    if allow_offset and kind < 0.18:
        unit = rng.choice(["m", "m", "h", "d"])
        if unit == "m":
            n = rng.choice([5, 10, 15, 20, 25, 30, 40, 45, 50, 90])
            if n == 30 and rng.random() < 0.4:
                return TimeExpr("yarım saat sonra", "+30m")
            return TimeExpr(f"{n} dakika sonra", f"+{n}m")
        if unit == "h":
            n = rng.randint(1, 12)
            return TimeExpr(f"{n} saat sonra", f"+{n}h")
        n = rng.randint(2, 14)
        return TimeExpr(f"{n} gün sonra", f"+{n}d")
    if kind < 0.45:
        return TimeExpr(f"yarın {clock}", f"d+1@{hm}")
    if kind < 0.58:
        return TimeExpr(f"bugün {clock}", f"d+0@{hm}")
    if kind < 0.64:
        return TimeExpr(f"öbür gün {clock}", f"d+2@{hm}")
    if kind < 0.7:
        n = rng.randint(3, 9)
        return TimeExpr(f"{n} gün sonra {clock}", f"d+{n}@{hm}")
    if kind < 0.88:
        tr, code = rng.choice(DOW_TR)
        return TimeExpr(f"{tr} {clock}", f"{code}@{hm}")
    mo = rng.randint(1, 12)
    d = rng.randint(1, MONTH_DAYS[mo - 1])
    return TimeExpr(f"{d} {MONTHS_TR[mo - 1]} {clock}", f"{mo:02d}-{d:02d}@{hm}")


# --------------------------------------------------------------------------- değerler
# Her listenin son ~%25'i test_iid için ayrılır (eğitimde görülmez).

REMINDER_TEXTS = [
    "toplantı", "ilaç içmek", "annemi aramak", "faturayı ödemek", "spor salonu", "su içmek",
    "çöpü çıkarmak", "kitap okumak", "doktor randevusu", "kargo teslimi", "sunum provası",
    "ders çalışmak", "markete gitmek", "arabanın muayenesi", "kira ödemesi", "çiçekleri sulamak",
    "yürüyüş", "e-postaları kontrol etmek", "babamı aramak", "vitamin almak", "proje teslimi",
    "kediyi beslemek", "diş fırçalamak", "toplantı notlarını göndermek", "kursa gitmek",
    "eczaneye uğramak", "banka randevusu", "yoga", "uyumak", "mola vermek", "dişçi",
    "sınav", "elektrik faturası", "doğum günü hediyesi", "kuaför", "veteriner randevusu",
    "çamaşırları asmak", "bulaşık makinesi", "okul toplantısı", "maç",
]
NOTE_TEXTS = [
    "süt ve ekmek al", "yeni proje fikri: telefonla çalışan ajan ekibi", "kitap önerisi: Sapiens",
    "toplantıda bütçe konuşulacak", "arabanın lastikleri değişecek", "film listesi: Interstellar",
    "spor programı: pazartesi koşu, çarşamba ağırlık", "tatil için Kaş'a bakılacak",
    "anneme hediye olarak atkı", "haftalık harcama 1200 lira", "bahçeye domates ekilecek",
    "telefonun yedeği alınacak", "kursun ödevi perşembeye", "yeni tarif: mercimek çorbası",
    "Orion için kart fikirleri", "evin kirası arttı", "dişçi kontrolü altı ayda bir",
    "kahve çekirdeği bitti", "sunum slaytları hazır", "kitaplığa raf eklenecek",
    "aylık abonelikleri gözden geçir", "bisikletin zinciri yağlanacak", "haftaya misafir gelecek",
    "yeni dil öğrenme planı", "su faturası yüksek geldi", "pazar günü kahvaltı", "kamera lensi araştır",
    "doğum günü listesi", "laptop pili zayıfladı", "piknik malzemeleri",
]
TAGS = ["iş", "ev", "alışveriş", "fikir", "sağlık", "okul", "proje", "aile", "para", "kitap"]
QUERIES = [
    "bütçe", "toplantı", "kitap", "tatil", "proje", "harcama", "tarif", "spor", "hediye", "araba",
    "fatura", "film", "kira", "yedek", "kurs", "kahve", "abonelik", "misafir", "bisiklet", "lens",
]
TOPICS = [
    "rakip fiyatları", "elektrikli araçlar", "yapay zeka ajanları", "Xiaomi Note 13 Pro güncellemeleri",
    "ücretsiz LLM API'leri", "İstanbul'da kiralık daireler", "Python ile veri analizi", "güneş paneli maliyeti",
    "uyku düzeni", "Kotlin coroutines", "telefon pil ömrü", "en iyi not uygulamaları", "kripto vergileri",
    "akdeniz diyeti", "evde kahve demleme", "bulut depolama fiyatları", "Android güvenlik ayarları",
    "çocuklar için kodlama", "uzaktan çalışma araçları", "yerel yapay zeka modelleri", "kamp ekipmanı",
    "kitap özetleme yöntemleri", "akıllı ev cihazları", "fotoğrafçılık temelleri",
]
REPOS = [
    "Eskay9633/Jarvis-", "torvalds/linux", "google/guava", "pytorch/pytorch", "ggml-org/llama.cpp",
    "square/okhttp", "JetBrains/kotlin", "android/nowinandroid", "openai/gpt-oss", "huggingface/transformers",
    "Kotlin/kotlinx.coroutines", "microsoft/vscode", "facebook/react", "tensorflow/tensorflow",
]
EVENT_TITLES = [
    "dişçi", "proje toplantısı", "yoga dersi", "akşam yemeği", "veli toplantısı", "spor", "sunum",
    "doktor kontrolü", "kahve buluşması", "araba servisi", "kurs", "sinema", "konser", "mülakat",
    "kuaför", "banka görüşmesi", "doğum günü partisi", "maç", "ders", "staj görüşmesi",
]
PLACES = ["Kadıköy", "ofis", "evde", "Beşiktaş", "spor salonu", "kafe", "okul", "hastane", "online", "Moda"]
CARD_TITLES = [
    "Haftalık harcama", "Su tüketimi", "Okunan kitaplar", "Günlük adım", "Proje ilerlemesi",
    "Yapılacaklar", "Kota durumu", "Sabah rutini", "Aylık bütçe", "Uyku süresi", "Ders saatleri",
    "Rakip fiyatları", "Toplantı notları", "Pil sağlığı", "Antrenman", "Haftalık rapor",
]
TIMER_LABELS = ["çay", "makarna", "çalışma", "mola", "egzersiz", "yumurta", "toplantı", "kitap okuma", "meditasyon"]
NEEDS = ["kod yazma", "Türkçe çeviri", "resim tanıma", "uzun metin özeti", "araştırma", "matematik"]
MEMORY_TEXTS = [
    "sabahları kahve yerine çay içerim", "raporlar kısa olsun", "toplantılar 10'dan önce olmasın",
    "hafta sonu bildirim istemiyorum", "notları madde madde yaz", "faturayı her ayın 5'inde öde",
    "kodları Kotlin ile yaz", "araştırmalarda en az üç kaynak kullan", "cevaplar Türkçe olsun",
    "Sophia önce ücretsiz kaynaklara baksın", "akşam 10'dan sonra rahatsız etme",
]


_LEX_CACHE: dict = {}


def _lex() -> dict:
    if not _LEX_CACHE:
        from .lexicon import NOUNS, VERBS, noun_forms, verb_forms
        _LEX_CACHE["nouns"] = [f for w in NOUNS for f in noun_forms(w)]
        _LEX_CACHE["inf"] = [verb_forms(v)[1] for v in VERBS if "_" not in v]
    return _LEX_CACHE


NAMES = ["Ayşe", "Mehmet", "Zeynep", "Ali", "Elif", "Can", "Deniz", "Ece", "Burak", "Selin", "Kerem", "Defne",
         "Emre", "Melis", "Ozan", "İrem", "Tolga", "Nehir", "Umut", "Yasemin"]
ADJ = ["yeni", "eski", "büyük", "küçük", "hızlı", "önemli", "acil", "haftalık", "aylık", "günlük", "uzun", "kısa",
       "kırmızı", "mavi", "ucuz", "pahalı", "kolay", "zor", "son", "ilk", "özel", "ortak", "yerel", "dijital"]


def random_phrase(rng: random.Random, lo: int = 1, hi: int = 4, title: bool = False) -> str:
    """Kopyalamayı öğretmek için rastgele (anlamsız olabilen) Türkçe ifade."""
    lx = _lex()
    n = rng.randint(lo, hi)
    words = []
    for i in range(n):
        r = rng.random()
        if r < 0.5:
            words.append(rng.choice(lx["nouns"]))
        elif r < 0.65:
            words.append(rng.choice(ADJ))
        elif r < 0.75:
            words.append(rng.choice(NAMES) + rng.choice(["", "", "'nin", "'yı", "'ya", "'la"]))
        elif r < 0.83:
            words.append(str(rng.choice([rng.randint(1, 31), rng.randint(100, 5000), rng.randint(2020, 2030)])))
        else:
            words.append(rng.choice(lx["inf"]) if i == n - 1 else rng.choice(lx["nouns"]))
    out = " ".join(words)
    if title:
        out = out[:1].upper() + out[1:]
    return out


def random_repo(rng: random.Random) -> str:
    alnum = "abcdefghijklmnopqrstuvwxyz0123456789"
    owner = rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") + "".join(
        rng.choice(alnum + "ABCDEFGHIJKLMNOPQRSTUVWXYZ-") for _ in range(rng.randint(2, 12)))
    owner = owner.rstrip("-")
    name = "".join(rng.choice(alnum + "-._") for _ in range(rng.randint(2, 16))).strip("-._") or "repo"
    return f"{owner}/{name}"


def pick(rng: random.Random, values: list, split: str, rand=None, p_rand: float = 0.7) -> str:
    """Eğitimde çoğunlukla rastgele ifade (kopyalama öğrenilsin), testte gerçekçi listeden."""
    if split == "train" and rand is not None and rng.random() < p_rand:
        v = rand(rng)
        while len(v) < 3 or len(v) > 75:
            v = rand(rng)
        return v
    return rng.choice(_split(values, split))


def _split(values: list, split: str) -> list:
    cut = max(1, int(len(values) * 0.75))
    return values[:cut] if split == "train" else values[cut:] if split == "test_iid" else values


POLITE_PRE = ["", "", "", "", "Orion, ", "lütfen ", "hey Orion ", "acaba "]
POLITE_POST = ["", "", "", "", " lütfen", " teşekkürler", " rica etsem", "."]


# --------------------------------------------------------------------------- şablonlar
# Her komut: [(şablon üretici, ...)]. Listenin sonundaki OOD_N şablon test_ood'a ayrılır.

Gen = Callable[[random.Random, str], tuple[str, dict]]
OOD_N = 2


def _rem(rng, sp):
    t = random_time(rng)
    txt = pick(rng, REMINDER_TEXTS, sp, lambda r: random_phrase(r, 1, 4))
    return t, txt


def reminder_templates() -> list[Gen]:
    def a(rng, sp):
        t, x = _rem(rng, sp); return f"{t.tr} {x} hatırlat", {"at": t.when, "text": x}
    def b(rng, sp):
        t, x = _rem(rng, sp); return f"{t.tr} bana {x} diye hatırlat", {"at": t.when, "text": x}
    def c(rng, sp):
        t, x = _rem(rng, sp); return f"{x} için {t.tr} hatırlatıcı kur", {"at": t.when, "text": x}
    def d(rng, sp):
        t, x = _rem(rng, sp); return f"hatırlatıcı kur: {x}, {t.tr}", {"at": t.when, "text": x}
    def e(rng, sp):
        h = rng.choice([7, 8, 9, 12, 13, 20, 21, 22]); m = rng.choice([0, 0, 30])
        x = pick(rng, REMINDER_TEXTS, sp, lambda r: random_phrase(r, 1, 4))
        return (f"her gün {_clock_phrase(rng, h, m)} {x} hatırlat",
                {"at": f"d+0@{h:02d}:{m:02d}", "text": x, "repeat": "daily"})
    def f(rng, sp):
        tr, code = rng.choice(DOW_TR); h = rng.choice([8, 9, 10, 18, 19]); x = pick(rng, REMINDER_TEXTS, sp, lambda r: random_phrase(r, 1, 4))
        return (f"her {tr} {_clock_phrase(rng, h, 0)} {x} hatırlat",
                {"at": f"{code}@{h:02d}:00", "text": x, "repeat": "weekly"})
    # OOD
    def g(rng, sp):
        t, x = _rem(rng, sp); return f"{x} konusunda {t.tr} beni uyar", {"at": t.when, "text": x}
    def h(rng, sp):
        t, x = _rem(rng, sp); return f"{t.tr} için alarm kur, {x}", {"at": t.when, "text": x}
    return [a, b, c, d, e, f, g, h]


def reminder_list_templates() -> list[Gen]:
    phrases = ["hatırlatıcılarımı göster", "bekleyen hatırlatmalar neler", "hangi hatırlatıcılar kurulu",
               "hatırlatıcıları listele", "kurulu alarmlarım neler", "hatırlatmalarıma bak",
               "ne hatırlatacaksın bana", "yaklaşan hatırlatıcılar hangileri"]
    return [(lambda p: (lambda rng, sp: (p, {})))(p) for p in phrases]


def note_write_templates() -> list[Gen]:
    def pick_(rng, sp): return pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 2, 9))
    def a(rng, sp): x = pick_(rng, sp); return f"not al: {x}", {"text": x}
    def b(rng, sp): x = pick_(rng, sp); return f"şunu not et: {x}", {"text": x}
    def c(rng, sp): x = pick_(rng, sp); return f"{x} diye not yaz", {"text": x}
    def d(rng, sp): x = pick_(rng, sp); return f"notlarıma ekle: {x}", {"text": x}
    def e(rng, sp):
        x = pick_(rng, sp); tg = rng.choice(TAGS); return f"#{tg} etiketiyle not al: {x}", {"text": x, "tags": [tg]}
    def f(rng, sp):
        x = pick_(rng, sp); t1, t2 = rng.sample(TAGS, 2)
        return f"not yaz, etiketler {t1} ve {t2}: {x}", {"text": x, "tags": [t1, t2]}
    def g(rng, sp): x = pick_(rng, sp); return f"bir kenara yaz: {x}", {"text": x}
    def h(rng, sp): x = pick_(rng, sp); return f"unutmamak için kaydet: {x}", {"text": x}
    return [a, b, c, d, e, f, g, h]


def note_search_templates() -> list[Gen]:
    def q(rng, sp): return pick(rng, QUERIES, sp, lambda r: random_phrase(r, 1, 2))
    def a(rng, sp): x = q(rng, sp); return f"notlarımda {x} ara", {"query": x}
    def b(rng, sp): x = q(rng, sp); return f"{x} ile ilgili notları bul", {"query": x}
    def c(rng, sp): x = q(rng, sp); return f"{x} hakkında not var mı", {"query": x}
    def d(rng, sp):
        x = q(rng, sp); n = rng.choice([3, 5, 10, 20]); return f"{x} geçen son {n} notu göster", {"query": x, "limit": n}
    def e(rng, sp): x = q(rng, sp); return f"notlarda {x} geçenleri listele", {"query": x}
    def f(rng, sp): x = q(rng, sp); return f"{x} diye bir not yazmış mıydım", {"query": x}
    def g(rng, sp): x = q(rng, sp); return f"{x} notlarını getir", {"query": x}
    return [a, b, c, d, e, f, g]


def research_templates() -> list[Gen]:
    def tp(rng, sp): return pick(rng, TOPICS, sp, lambda r: random_phrase(r, 1, 5))
    def a(rng, sp): x = tp(rng, sp); return f"{x} hakkında araştırma yap", {"topic": x}
    def b(rng, sp): x = tp(rng, sp); return f"{x} konusunu araştır", {"topic": x}
    def c(rng, sp): x = tp(rng, sp); return f"hızlıca {x} araştır", {"topic": x, "depth": "quick"}
    def d(rng, sp): x = tp(rng, sp); return f"{x} hakkında derin araştırma yap", {"topic": x, "depth": "deep"}
    def e(rng, sp): x = tp(rng, sp); return f"{x} için detaylı bir araştırma başlat", {"topic": x, "depth": "deep"}
    def f(rng, sp): x = tp(rng, sp); return f"kısaca {x} nedir araştır", {"topic": x, "depth": "quick"}
    def g(rng, sp): x = tp(rng, sp); return f"Sophia {x} konusuna baksın", {"topic": x}
    def h(rng, sp): x = tp(rng, sp); return f"{x} hakkında ne bulabilirsin", {"topic": x}
    return [a, b, c, d, e, f, g, h]


def repo_templates() -> list[Gen]:
    def r(rng, sp): return pick(rng, REPOS, sp, random_repo, 0.85)
    def a(rng, sp): x = r(rng, sp); return f"{x} reposunu denetle", {"repo": x}
    def b(rng, sp): x = r(rng, sp); return f"{x} deposunu kontrol et", {"repo": x}
    def c(rng, sp): x = r(rng, sp); return f"{x} CI durumuna bak", {"repo": x, "scope": "ci"}
    def d(rng, sp): x = r(rng, sp); return f"{x} için güvenlik denetimi yap", {"repo": x, "scope": "security"}
    def e(rng, sp): x = r(rng, sp); return f"{x} derlemesi geçti mi", {"repo": x, "scope": "ci"}
    def f(rng, sp): x = r(rng, sp); return f"Jack {x} reposuna göz atsın", {"repo": x}
    def g(rng, sp): x = r(rng, sp); return f"{x} projesinin son durumunu özetle", {"repo": x}
    return [a, b, c, d, e, f, g]


def calendar_add_templates() -> list[Gen]:
    def ev(rng, sp):
        return random_time(rng, allow_offset=False), pick(rng, EVENT_TITLES, sp, lambda r: random_phrase(r, 1, 3))
    def a(rng, sp): t, x = ev(rng, sp); return f"{t.tr} takvime {x} ekle", {"title": x, "start": t.when}
    def b(rng, sp): t, x = ev(rng, sp); return f"takvimime {t.tr} {x} koy", {"title": x, "start": t.when}
    def c(rng, sp):
        t, x = ev(rng, sp); n = rng.choice([15, 30, 45, 90, 120])
        return f"{t.tr} {x} etkinliği oluştur, {n} dakika sürecek", {"title": x, "start": t.when, "minutes": n}
    def d(rng, sp):
        t, x = ev(rng, sp); p = pick(rng, PLACES, sp, lambda r: random_phrase(r, 1, 2, True), 0.5)
        return f"{t.tr} {x} var, yer {p}, takvime ekle", {"title": x, "start": t.when, "place": p}
    def e(rng, sp):
        t, x = ev(rng, sp); n = rng.choice([1, 2, 3])
        return f"{t.tr} {n} saatlik {x} planla", {"title": x, "start": t.when, "minutes": n * 60}
    def f(rng, sp): t, x = ev(rng, sp); return f"{x} için {t.tr} randevu oluştur", {"title": x, "start": t.when}
    def g(rng, sp): t, x = ev(rng, sp); return f"ajandama yaz: {x}, {t.tr}", {"title": x, "start": t.when}
    return [a, b, c, d, e, f, g]


def calendar_list_templates() -> list[Gen]:
    days = [("bugün", "d+0"), ("yarın", "d+1"), ("öbür gün", "d+2")]
    def a(rng, sp): tr, d = rng.choice(days); return f"{tr} takvimde ne var", {"day": d}
    def b(rng, sp): tr, d = rng.choice(days); return f"{tr} programım ne", {"day": d}
    def c(rng, sp): n = rng.randint(3, 9); return f"{n} gün sonra neler var", {"day": f"d+{n}"}
    def d(rng, sp): tr, d = rng.choice(days); return f"{tr} için etkinliklerimi göster", {"day": d}
    def e(rng, sp): tr, d = rng.choice(days); return f"{tr} neler yapacağım", {"day": d}
    def f(rng, sp): tr, d = rng.choice(days); return f"{tr} boş muyum, takvime bak", {"day": d}
    return [a, b, c, d, e, f]


ATTACHED = "content://orion/ek"


def file_templates() -> list[Gen]:
    def a(rng, sp): return "ekteki dosyayı özetle", {"uri": ATTACHED}
    def b(rng, sp): return "bu dosyayı kısaca özetle", {"uri": ATTACHED, "length": "short"}
    def c(rng, sp): return "eklediğim belgenin uzun özetini çıkar", {"uri": ATTACHED, "length": "long"}
    def d(rng, sp): return "dosyanın orta uzunlukta özetini yaz", {"uri": ATTACHED, "length": "medium"}
    def e(rng, sp): return "bu belgede ne yazıyor, özetle", {"uri": ATTACHED}
    def f(rng, sp): return "eki okuyup ayrıntılı özet ver", {"uri": ATTACHED, "length": "long"}
    return [a, b, c, d, e, f]


IMAGE_PROMPTS = [
    "gün batımında İstanbul boğazı", "uzayda yüzen bir kedi", "karlı dağlarda ahşap bir kulübe", "robot bir aşçı",
    "su altında bir şehir", "Orion takımyıldızı altında kamp", "pastel renkli bir kahve dükkanı", "çizgi film tarzı bir ejderha",
    "minimal bir uygulama simgesi", "yağmurlu bir Kadıköy sokağı", "fütüristik bir telefon", "çiçek açmış bir kiraz ağacı",
    "eski bir kütüphane", "neon ışıklı bir oyun salonu", "deniz kenarında bir deniz feneri", "sevimli bir baykuş",
]
QUESTIONS = [
    "bu hata nasıl düzelir", "burada ne yazıyor", "hangi ayarı değiştirmeliyim", "fiyat ne kadar", "bu uyarı ne demek",
    "toplamı hesapla", "bu kod neden çalışmıyor", "tarih ne zaman", "hangi butona basmalıyım", "özet çıkar",
]


def image_analyze_templates() -> list[Gen]:
    def q(rng, sp): return pick(rng, QUESTIONS, sp, lambda r: random_phrase(r, 2, 6))
    def a(rng, sp): return "ekteki fotoğrafı oku", {"uri": ATTACHED}
    def b(rng, sp): return "bu ekran görüntüsünde ne yazıyor", {"uri": ATTACHED}
    def c(rng, sp): x = q(rng, sp); return f"resmi incele: {x}", {"uri": ATTACHED, "question": x}
    def d(rng, sp): x = q(rng, sp); return f"ekran görüntüsüne bak ve şunu cevapla: {x}", {"uri": ATTACHED, "question": x}
    def e(rng, sp): x = q(rng, sp); return f"fotoğrafa bakıp söyle: {x}", {"uri": ATTACHED, "question": x}
    def f(rng, sp): return "ss'teki sorunu çöz", {"uri": ATTACHED, "question": "sorunu çöz"}
    def g(rng, sp): return "fotoğraftaki yazıyı çıkar", {"uri": ATTACHED}
    def h(rng, sp): x = q(rng, sp); return f"görseli açıkla, {x}", {"uri": ATTACHED, "question": x}
    return [a, b, c, d, e, f, g, h]


def image_generate_templates() -> list[Gen]:
    def p(rng, sp): return pick(rng, IMAGE_PROMPTS, sp, lambda r: random_phrase(r, 2, 6))
    def a(rng, sp): x = p(rng, sp); return f"{x} resmi çiz", {"prompt": x}
    def b(rng, sp): x = p(rng, sp); return f"bana {x} görseli oluştur", {"prompt": x}
    def c(rng, sp): x = p(rng, sp); return f"resim oluştur: {x}", {"prompt": x}
    def d(rng, sp): x = p(rng, sp); return f"dikey bir {x} resmi yap", {"prompt": x, "size": "portrait"}
    def e(rng, sp): x = p(rng, sp); return f"yatay {x} görseli üret", {"prompt": x, "size": "landscape"}
    def f(rng, sp): x = p(rng, sp); return f"görsel üret: {x}", {"prompt": x}
    def g(rng, sp): x = p(rng, sp); return f"{x} temalı bir illüstrasyon hazırla", {"prompt": x}
    def h(rng, sp): x = p(rng, sp); return f"Luna {x} resmi tasarlasın", {"prompt": x}
    return [a, b, c, d, e, f, g, h]


QUESTIONS_GENERAL = [
    "kara delik nedir", "fotosentez nasıl olur", "İstanbul'un nüfusu kaç", "Python ile Kotlin farkı ne",
    "iyi uyumak için ne yapmalıyım", "ekmek nasıl mayalanır", "elektrikli araba mı benzinli mi", "Osmanlı ne zaman kuruldu",
    "kredi kartı faizi nasıl hesaplanır", "telefonum neden ısınıyor", "en iyi kahve demleme yöntemi ne", "bulut depolama nedir",
]
LANGS = [("İngilizce", "en"), ("Almanca", "de"), ("Fransızca", "fr"), ("İspanyolca", "es"), ("Arapça", "ar"), ("Rusça", "ru"),
         ("İtalyanca", "it"), ("Türkçe", "tr")]


def ask_templates() -> list[Gen]:
    def q(rng, sp): return pick(rng, QUESTIONS_GENERAL, sp, lambda r: random_phrase(r, 2, 6))
    def a(rng, sp): x = q(rng, sp); return f"{x}?", {"question": x}
    def b(rng, sp): x = q(rng, sp); return f"soru: {x}", {"question": x}
    def c(rng, sp): x = q(rng, sp); return f"bana anlat: {x}", {"question": x}
    def d(rng, sp): x = q(rng, sp); return f"merak ediyorum, {x}", {"question": x}
    def e(rng, sp): x = q(rng, sp); return f"Orion'a sor: {x}", {"question": x}
    def f(rng, sp): x = q(rng, sp); return f"açıkla: {x}", {"question": x}
    return [a, b, c, d, e, f]


def calc_templates() -> list[Gen]:
    def n(rng): return rng.choice([2, 3, 5, 12, 18, 25, 40, 75, 100, 150, 1250, 3600])
    def a(rng, sp): x, y = n(rng), n(rng); return f"{x} çarpı {y} kaç", {"expr": f"{x}*{y}"}
    def b(rng, sp): x, y = n(rng), n(rng); return f"{x} artı {y} kaç eder", {"expr": f"{x}+{y}"}
    def c(rng, sp): x, y = n(rng), n(rng); return f"hesapla: {x} bölü {y}", {"expr": f"{x}/{y}"}
    def d(rng, sp): x, y = n(rng), n(rng); return f"{x}'in yüzde {y}'si kaç", {"expr": f"{x}*{y}/100"}
    def e(rng, sp): x, y = n(rng), n(rng); return f"{x} eksi {y} kaç", {"expr": f"{x}-{y}"}
    def f(rng, sp): x, y, z = n(rng), n(rng), n(rng); return f"hesapla: ({x}+{y})*{z}", {"expr": f"({x}+{y})*{z}"}
    return [a, b, c, d, e, f]


def translate_templates() -> list[Gen]:
    def t(rng, sp): return pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 2, 8))
    def a(rng, sp): x = t(rng, sp); ln, lc = rng.choice(LANGS); return f"{ln}ye çevir: {x}", {"text": x, "to": lc}
    def b(rng, sp): x = t(rng, sp); ln, lc = rng.choice(LANGS); return f"şunu {ln} yap: {x}", {"text": x, "to": lc}
    def c(rng, sp): x = t(rng, sp); return f"çevir: {x}", {"text": x}
    def d(rng, sp): x = t(rng, sp); ln, lc = rng.choice(LANGS); return f"{ln} karşılığı ne: {x}", {"text": x, "to": lc}
    def e(rng, sp): x = t(rng, sp); ln, lc = rng.choice(LANGS); return f"Emma {ln}ye çevirsin: {x}", {"text": x, "to": lc}
    def f(rng, sp): x = t(rng, sp); ln, lc = rng.choice(LANGS); return f"tercüme et ({ln}): {x}", {"text": x, "to": lc}
    return [a, b, c, d, e, f]


CARD_KIND_WORDS = {
    "chart": ["grafik", "çubuk grafik", "çizgi grafik"], "list": ["liste", "yapılacaklar listesi"],
    "timer": ["zamanlayıcı", "geri sayım"], "metric": ["metrik", "sayı"], "note": ["not"],
    "reminder": ["hatırlatıcı"], "research": ["araştırma"], "approval": ["onay"],
}


def card_templates() -> list[Gen]:
    def kt(rng, sp):
        kind = rng.choice(list(CARD_KIND_WORDS)); return kind, rng.choice(CARD_KIND_WORDS[kind]), pick(rng, CARD_TITLES, sp, lambda r: random_phrase(r, 1, 3, True))
    def a(rng, sp): k, w, t = kt(rng, sp); return f"'{t}' başlıklı bir {w} kartı ekle", {"kind": k, "title": t}
    def b(rng, sp): k, w, t = kt(rng, sp); return f"{w} kartı ekle: {t}", {"kind": k, "title": t}
    def c(rng, sp):
        k, w, t = kt(rng, sp); r, rtr = rng.choice([("daily", "her gün"), ("weekly", "her hafta"), ("hourly", "her saat")])
        return f"{t} için {w} kartı yap, {rtr} yenilensin", {"kind": k, "title": t, "refresh": r}
    def d(rng, sp): k, w, t = kt(rng, sp); return f"ana ekrana büyük bir {w} kartı koy: {t}", {"kind": k, "title": t, "size": "full"}
    def e(rng, sp):
        t = pick(rng, CARD_TITLES, sp, lambda r: random_phrase(r, 1, 3, True)); s, s_tr = rng.choice([("notes", "notlardan"), ("calendar", "takvimden"), ("quota", "kotadan")])
        return f"{s_tr} beslenen bir grafik kartı ekle: {t}", {"kind": "chart", "title": t, "source": s}
    def f(rng, sp): k, w, t = kt(rng, sp); return f"Luna {t} için bir {w} kartı tasarlasın", {"kind": k, "title": t}
    def g(rng, sp): k, w, t = kt(rng, sp); return f"ekranıma {t} gösteren {w} kartı ekler misin", {"kind": k, "title": t}
    return [a, b, c, d, e, f, g]


def timer_templates() -> list[Gen]:
    def n(rng): return rng.choice([1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 45, 60, 90])
    def a(rng, sp): m = n(rng); return f"{m} dakikalık zamanlayıcı kur", {"minutes": m}
    def b(rng, sp): m = n(rng); return f"{m} dakika geri sayım başlat", {"minutes": m}
    def c(rng, sp): m = n(rng); lb = pick(rng, TIMER_LABELS, sp, lambda r: random_phrase(r, 1, 2)); return f"{lb} için {m} dakika say", {"minutes": m, "label": lb}
    def d(rng, sp): m = n(rng); lb = pick(rng, TIMER_LABELS, sp, lambda r: random_phrase(r, 1, 2)); return f"{lb} zamanlayıcısı, {m} dakika", {"minutes": m, "label": lb}
    def e(rng, sp): m = n(rng); return f"{m} dakika sonra haber ver diye sayaç başlat", {"minutes": m}
    def f(rng, sp): m = n(rng); lb = pick(rng, TIMER_LABELS, sp, lambda r: random_phrase(r, 1, 2)); return f"{m} dakikalık {lb} sayacı aç", {"minutes": m, "label": lb}
    return [a, b, c, d, e, f]


def discover_templates() -> list[Gen]:
    def a(rng, sp): return "yeni ücretsiz yapay zeka bul", {}
    def b(rng, sp): return "başka ücretsiz model var mı araştır", {}
    def c(rng, sp): x = pick(rng, NEEDS, sp, lambda r: random_phrase(r, 1, 3)); return f"{x} için ücretsiz yapay zeka bul", {"need": x}
    def d(rng, sp): return "beyne eklenebilecek yeni sağlayıcı ara", {}
    def e(rng, sp): x = pick(rng, NEEDS, sp, lambda r: random_phrase(r, 1, 3)); return f"{x} yapabilen bedava bir model öner", {"need": x}
    def f(rng, sp): return "kotası dolmayan yeni bir API bul", {}
    return [a, b, c, d, e, f]


def health_templates() -> list[Gen]:
    def a(rng, sp): return "telefonun sağlığını kontrol et", {}
    def b(rng, sp): return "güncellemeden önce telefonu kontrol et", {"purpose": "update"}
    def c(rng, sp): return "pil ve depolama durumu nasıl", {}
    def d(rng, sp): return "güncelleme yapsam sorun olur mu", {"purpose": "update"}
    def e(rng, sp): return "Maya telefonu bir ölçsün", {}
    def f(rng, sp): return "sistem güncellemesine hazır mıyım", {"purpose": "update"}
    return [a, b, c, d, e, f]


def memory_templates() -> list[Gen]:
    def x(rng, sp): return pick(rng, MEMORY_TEXTS, sp, lambda r: random_phrase(r, 2, 7))
    def a(rng, sp): t = x(rng, sp); return f"bunu hatırla: {t}", {"kind": "note", "text": t}
    def b(rng, sp): t = x(rng, sp); return f"tercihim şu: {t}", {"kind": "preference", "text": t}
    def c(rng, sp): t = x(rng, sp); return f"bu adımları beceri olarak kaydet: {t}", {"kind": "skill", "text": t}
    def d(rng, sp): t = x(rng, sp); return f"şunu aklında tut: {t}", {"kind": "note", "text": t}
    def e(rng, sp): t = x(rng, sp); return f"benim için kural: {t}", {"kind": "preference", "text": t}
    def f(rng, sp): t = x(rng, sp); return f"kalıcı hafızaya yaz: {t}", {"kind": "note", "text": t}
    return [a, b, c, d, e, f]


def help_templates() -> list[Gen]:
    phrases = ["neler yapabilirsin", "yardım", "komutları göster", "ne işe yararsın", "bana neler yaparsın",
               "hangi işleri yapabiliyorsun", "nasıl kullanılıyorsun", "yardım menüsü"]
    return [(lambda p: (lambda rng, sp: (p, {})))(p) for p in phrases]


TEMPLATES: dict[str, list[Gen]] = {
    "reminder.set": reminder_templates(),
    "reminder.list": reminder_list_templates(),
    "note.write": note_write_templates(),
    "note.search": note_search_templates(),
    "research.start": research_templates(),
    "repo.audit": repo_templates(),
    "calendar.add": calendar_add_templates(),
    "calendar.list": calendar_list_templates(),
    "file.summarize": file_templates(),
    "image.analyze": image_analyze_templates(),
    "image.generate": image_generate_templates(),
    "card.propose": card_templates(),
    "timer.start": timer_templates(),
    "provider.discover": discover_templates(),
    "health.check": health_templates(),
    "memory.save": memory_templates(),
    "ask": ask_templates(),
    "calc.eval": calc_templates(),
    "text.translate": translate_templates(),
    "help": help_templates(),
    "market.crypto": [lambda rng, sp: g_crypto(rng, sp, False, False)] * 3,
    "skill.run": [lambda rng, sp: g_skill_run(rng, sp, False, False)] * 3,
    "skill.list": [lambda rng, sp: g_skill_list(rng, sp, False, False)] * 3,
    "skill.learn": [lambda rng, sp: g_skill_learn(rng, sp, False, False)] * 3,
    "skill.record": [lambda rng, sp: g_skill_record(rng, sp, False, False)] * 3,
    "agent.hire": [lambda rng, sp: g_agent_hire(rng, sp, False, False)] * 3,
    "mail.check": [lambda rng, sp: g_mail_check(rng, sp, False, False)] * 3,
    "mail.send": [lambda rng, sp: g_mail_send(rng, sp, False, False)] * 3,
}

# Komut ağırlıkları (gerçek kullanım tahmini; çok değerli olanlar daha sık)
WEIGHTS = {
    "reminder.set": 6, "reminder.list": 1, "note.write": 4, "note.search": 3, "research.start": 3,
    "repo.audit": 2, "calendar.add": 4, "calendar.list": 2, "file.summarize": 1, "image.analyze": 2, "image.generate": 2, "card.propose": 3,
    "timer.start": 2, "provider.discover": 1, "health.check": 1, "memory.save": 2, "ask": 3, "calc.eval": 2, "text.translate": 2, "market.crypto": 2, "skill.run": 2, "skill.list": 1, "skill.learn": 2, "skill.record": 1, "agent.hire": 1, "mail.check": 2, "mail.send": 1, "help": 1,
}


def _templates_for(cmd: str, split: str) -> list[Gen]:
    t = TEMPLATES[cmd]
    if len(t) <= OOD_N + 2:
        return t  # çok az şablon: ayırmaya değmez
    return t[-OOD_N:] if split == "test_ood" else t[:-OOD_N]


_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_FREE_TEXT_ARGS = {"text", "title", "topic", "query", "label", "place", "need"}


def _noise(rng: random.Random, s: str, args: dict) -> tuple[str, dict]:
    """Telefonda yazım: kesme işareti düşmesi, bazen Türkçe karakter kaybı.
    Karakter kaybında serbest metin argümanları da aynı biçimde değişir (model kopyalar, düzeltmez).
    Büyük harf normalleştirmesi uygulamada yapılır (Normalizer), burada üretilmez."""
    r = rng.random()
    if r < 0.08:
        s = s.replace("'", "")
    elif r < 0.12:
        s = s.translate(_ASCII)
        args = {k: (v.translate(_ASCII) if k in _FREE_TEXT_ARGS and isinstance(v, str) else v) for k, v in args.items()}
    return s, args


def t2c_example(rng: random.Random, split: str = "train", noise: bool = True) -> dict:
    """split: train | test_iid | test_ood (dilbilgisi motorunun görülmemiş parçaları) |
    legacy_ood (Tur 1-2'nin ayrılmış eski kalıpları; eğitimde hiç kullanılmaz, yalnızca rapor)."""
    cmds = list(WEIGHTS)
    cmd = rng.choices(cmds, weights=[WEIGHTS[c] for c in cmds])[0]
    if split == "legacy_ood":
        tr, args = rng.choice(_templates_for(cmd, "test_ood"))(rng, "test_ood")
    elif split == "test_ood" or rng.random() < GRAMMAR_FRAC:
        tr, args = grammar_example(rng, cmd, split)
    else:
        tr, args = rng.choice(_templates_for(cmd, split))(rng, split)
    tr = rng.choice(POLITE_PRE) + tr + rng.choice(POLITE_POST)
    if noise and split == "train":
        tr, args = _noise(rng, tr, args)
    req = validate_request({"v": 1, "id": "g", "cmd": cmd, "args": args})
    # Hedef: yalnız kullanıcının söylediği argümanlar (varsayılanlar doldurulmaz → kısa çıktı)
    return {"task": "t2c", "src": tr.strip(), "tgt": canonical({"cmd": cmd, "args": args}),
            "full": canonical(req)}


# --------------------------------------------------------------------------- plan

PLANS = {
    "reminder.set": ["when.resolve", "reminder.schedule", "memory.log"],
    "reminder.list": ["reminder.query", "card.render"],
    "note.write": ["secret.check", "note.insert", "memory.log"],
    "note.search": ["note.query", "card.render"],
    "research.start": ["net.check", "provider.ask", "sources.collect", "note.insert", "card.render"],
    "repo.audit": ["net.check", "github.read", "ci.read", "report.write", "card.render"],
    "calendar.add": ["when.resolve", "calendar.insert", "memory.log"],
    "calendar.list": ["calendar.query", "card.render"],
    "file.summarize": ["file.read", "secret.check", "summarize", "card.render"],
    "image.analyze": ["image.read", "ocr.run", "secret.check", "brain.reason", "card.render"],
    "image.generate": ["net.check", "image.provider", "image.save", "card.render"],
    "card.propose": ["card.validate", "proposal.create", "user.confirm"],
    "timer.start": ["timer.schedule", "card.render"],
    "provider.discover": ["net.check", "provider.ask", "provider.validate", "proposal.create", "user.confirm"],
    "health.check": ["battery.read", "storage.read", "backup.check", "card.render"],
    "memory.save": ["secret.check", "proposal.create", "user.confirm"],
    "ask": ["memory.recall", "brain.answer"],
    "calc.eval": ["expr.parse", "expr.eval"],
    "text.translate": ["brain.translate"],
    "market.crypto": ["net.check", "market.read", "card.render"],
    "skill.run": ["skill.load", "skill.validate", "steps.run", "card.render"],
    "skill.list": ["skill.load", "card.render"],
    "skill.learn": ["net.check", "skill.design", "skill.validate", "proposal.create", "user.confirm"],
    "skill.record": ["history.read", "skill.validate", "skill.save"],
    "agent.hire": ["agent.validate", "agent.save"],
    "mail.check": ["net.check", "vault.read", "imap.read", "secret.scrub", "card.render"],
    "mail.send": ["secret.check", "proposal.create", "user.confirm", "smtp.send"],
    "help": ["commands.list"],
}


def plan_for(req: dict) -> str:
    from .protocol import load_schema
    spec = load_schema().commands[req["cmd"]]
    steps = list(PLANS[req["cmd"]])
    if req["cmd"] == "research.start" and req["args"].get("depth") == "deep":
        steps.insert(3, "provider.judge")
    if req["cmd"] == "reminder.set" and req["args"].get("repeat", "none") != "none":
        steps.insert(2, "repeat.schedule")
    return canonical({"agent": spec["agent"], "steps": steps})


def plan_example(rng: random.Random, split: str = "train") -> dict:
    ex = t2c_example(rng, split, noise=False)
    req = json.loads(ex["tgt"])
    return {"task": "plan", "src": ex["tgt"], "tgt": plan_for(req)}


# --------------------------------------------------------------------------- cevap

ERR_TR = {
    "E_OFFLINE": "Bunun için internet gerekiyor; bağlantı yok.",
    "E_QUOTA": "Bugünlük ücretsiz kotalar doldu.",
    "E_PROVIDER": "Sağlayıcıdan cevap alamadım.",
    "E_NOT_FOUND": "Bulamadım.",
    "E_SECRET": "Gizli bilgi içerdiği için kaydetmedim.",
    "E_PERMISSION": "Bunun için izin yok.",
    "E_AMBIGUOUS": "Tam anlayamadım.",
}


def answer_example(rng: random.Random, split: str = "train") -> dict:
    """Cevap JSON'undan sabit biçimli Türkçe cevap (Kotlin Responder ile aynı kalıplar)."""
    kind = rng.random()
    if kind < 0.3:
        code = rng.choice(list(ERR_TR))
        resp = {"v": 1, "id": "g", "status": "error", "result": None, "err_code": code, "next": []}
        return {"task": "ans", "src": canonical(resp), "tgt": ERR_TR[code]}
    if kind < 0.55:
        n = rng.randint(0, 12)
        resp = {"v": 1, "id": "g", "status": "ok", "result": {"count": n, "kind": "note"}, "err_code": None, "next": []}
        tgt = "Hiç not bulamadım." if n == 0 else f"{n} not buldum."
        return {"task": "ans", "src": canonical(resp), "tgt": tgt}
    if kind < 0.8:
        x = rng.choice(_split(REMINDER_TEXTS, split))
        resp = {"v": 1, "id": "g", "status": "ok", "result": {"kind": "reminder", "text": x}, "err_code": None, "next": []}
        return {"task": "ans", "src": canonical(resp), "tgt": f"Tamam, \"{x}\" için hatırlatıcı kurdum."}
    resp = {"v": 1, "id": "g", "status": "needs_confirmation", "result": {"kind": "proposal"}, "err_code": None, "next": []}
    return {"task": "ans", "src": canonical(resp), "tgt": "Bunu yapmadan önce onayını bekliyorum."}


# --------------------------------------------------------------------------- toplu üretim

def generate(n: int, split: str, seed: int, mix: tuple[float, float, float] = (0.8, 0.12, 0.08)) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < mix[0]:
            out.append(t2c_example(rng, split))
        elif r < mix[0] + mix[1]:
            out.append(plan_example(rng, split))
        else:
            out.append(answer_example(rng, split))
    return out


def t2c_set(n: int, split: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [t2c_example(rng, split) for _ in range(n)]


def export_eval_fixture(path, n_iid: int = 600, n_ood: int = 400) -> None:
    """Kotlin çevirici testleri için sabit değerlendirme kümesi (Python ile aynı tohumlar)."""
    from pathlib import Path
    rows = []
    for split, n, seed in (("test_iid", n_iid, 101), ("test_ood", n_ood, 202)):
        for e in t2c_set(n, split, seed):
            rows.append({"split": split, "src": e["src"], "full": e["full"]})
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# --------------------------------------------------------------------------- dilbilgisi motoru (Tur 3)
# Kök sebep düzeltmesi: sabit 98 kalıp yerine parçalardan kurulan cümleler (bkz. grammar.py).

from . import grammar as G  # noqa: E402


def _frame(rng, sp, frames, hf):
    return G.choose(rng, frames, sp, hf)


def _verb(rng, sp, verbs, hv, mood=True):
    v = G.choose(rng, verbs, sp, hv)
    if mood and rng.random() < 0.35:
        return rng.choice(G.moods(v)[1:])
    return v


def _fill(frame: str, **kw) -> str:
    out = frame
    for k, v in kw.items():
        out = out.replace("{" + k + "}", v)
    return " ".join(out.split()).replace(" ,", ",")


REM_VERBS = ["hatırlat", "hatırlatıcı kur", "haber ver", "anımsat", "uyar", "hatırlatma kur", "alarm kur",
             "bildirim gönder", "hatırlatma ekle", "aklıma getir", "alarm ayarla", "beni uyar", "bana hatırlat",
             "hatırlatıcı ekle", "bildir", "hatırlatıcı oluştur"]
REM_FRAMES = ["{T} {X} {V}", "{T} bana {X} diye {V}", "{X} için {T} {V}", "{V}: {X}, {T}",
              "{T} {X} için {V}", "{X} diye {T} {V}", "{X}, {T}, {V}", "{X} konusunda {T} {V}",
              "{T} {V}, {X}", "{T} bana {X} için {V}", "{X} için bana {T} {V}", "{V}, {T} {X}"]
REM_ACC_FRAMES = ["{T} {XA} {V}", "{XA} {T} {V}", "{T} bana {XA} {V}", "{XA} unutmayayım, {T} {V}"]
REM_ACC_VERBS = ["hatırlat", "anımsat", "bana hatırlat", "aklıma getir", "unutturma", "hatırlatıver"]


def g_reminder(rng, sp, hf, hv):
    t = random_time(rng)
    if rng.random() < 0.3:
        inf = pick(rng, [x for x in REMINDER_TEXTS if G.inf_accusative(x)], sp,
                   lambda r: (random_phrase(r, 0, 2) + " " + r.choice(_lex()["inf"])).strip())
        acc = G.inf_accusative(inf)
        if acc:
            v = G.choose(rng, REM_ACC_VERBS, sp, hv)
            return _fill(_frame(rng, sp, REM_ACC_FRAMES, hf), T=t.tr, XA=acc, V=v), {"at": t.when, "text": inf}
    x = pick(rng, REMINDER_TEXTS, sp, lambda r: random_phrase(r, 1, 4))
    v = _verb(rng, sp, REM_VERBS, hv)
    return _fill(_frame(rng, sp, REM_FRAMES, hf), T=t.tr, X=x, V=v), {"at": t.when, "text": x}


REMLIST_SUBJ = ["hatırlatıcılarım", "hatırlatmalarım", "alarmlarım", "kurulu hatırlatıcılar", "bekleyen hatırlatmalar",
                "yaklaşan hatırlatıcılar", "kurduğun hatırlatıcılar", "hatırlatıcı listem"]
REMLIST_PRED = ["neler", "hangileri", "göster", "listele", "ne durumda", "nelerdi", "bir bak", "aç", "kaç tane",
                "neler var", "getir", "söyle"]


def g_reminder_list(rng, sp, hf, hv):
    s = G.choose(rng, REMLIST_SUBJ, sp, hf)
    p = G.choose(rng, REMLIST_PRED, sp, hv)
    if p in ("göster", "listele", "getir", "aç", "söyle") and rng.random() < 0.5:
        s = s + "ı" if s.endswith("lar") or s.endswith("ım") or s.endswith("ler") else s
    return f"{s} {p}", {}


NOTE_VERBS = ["not al", "not et", "not yaz", "notlarıma ekle", "bir kenara yaz", "kaydet", "yaz bir yere",
              "not defterime ekle", "deftere yaz", "not düş", "kayda geç", "not olarak kaydet", "bir yere not et",
              "notlara ekle", "not bırak", "yazıver"]
NOTE_FRAMES = ["{V}: {X}", "şunu {V}: {X}", "{X} diye {V}", "{X}, {V}", "{V} lütfen: {X}", "bunu {V}: {X}",
               "{V} — {X}", "hemen {V}: {X}"]


def g_note(rng, sp, hf, hv):
    x = pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 2, 9))
    v = _verb(rng, sp, NOTE_VERBS, hv)
    return _fill(_frame(rng, sp, NOTE_FRAMES, hf), X=x, V=v), {"text": x}


NS_VERBS = ["bul", "göster", "getir", "listele", "ara", "aç", "çıkar", "bir bak"]
NS_FRAMES = ["notlarımda {Q} {V}", "{Q} ile ilgili notları {V}", "{Q} geçen notları {V}", "notlarda {Q} geçenleri {V}",
             "{Q} notlarını {V}", "{Q} hakkındaki notlarımı {V}", "{Q} konulu notlarımı {V}", "notlarımdan {Q} olanları {V}"]
NS_QFRAMES = ["{Q} hakkında not var mı", "{Q} diye bir not yazmış mıydım", "{Q} ile ilgili ne not almıştım",
              "{Q} hakkında ne yazmıştım", "{Q} notum var mıydı", "{Q} ile ilgili notum var mı", "{Q} konusunda not almış mıydım",
              "notlarımda {Q} geçiyor mu"]


def g_note_search(rng, sp, hf, hv):
    q = pick(rng, QUERIES, sp, lambda r: random_phrase(r, 1, 2))
    if rng.random() < 0.4:
        return _fill(_frame(rng, sp, NS_QFRAMES, hf or hv), Q=q), {"query": q}
    return _fill(_frame(rng, sp, NS_FRAMES, hf), Q=q, V=_verb(rng, sp, NS_VERBS, hv)), {"query": q}


RES_VERBS = ["araştır", "araştırma yap", "incele", "hakkında bilgi topla", "araştırıp anlat", "bir araştır",
             "araştırma başlat", "derle", "bilgi bul", "kaynak tara", "araştırıver", "araştırma hazırla"]
RES_FRAMES = ["{X} hakkında {V}", "{X} konusunu {V}", "{V}: {X}", "{X} ile ilgili {V}", "{X} için {V}",
              "bana {X} {V}", "{X} konusunda {V}", "şu konuyu {V}: {X}"]
DEPTH = [("", None), ("", None), ("hızlıca", "quick"), ("kısaca", "quick"), ("derinlemesine", "deep"),
         ("detaylı", "deep"), ("ayrıntılı", "deep"), ("kısa bir", "quick")]


def g_research(rng, sp, hf, hv):
    x = pick(rng, TOPICS, sp, lambda r: random_phrase(r, 1, 5))
    v = G.choose(rng, RES_VERBS, sp, hv)
    if v in ("hakkında bilgi topla",):
        f = "{X} {V}"
    else:
        f = _frame(rng, sp, RES_FRAMES, hf)
    d_tr, d = rng.choice(DEPTH)
    if d_tr:
        f = f.replace("{V}", d_tr + " {V}", 1)
    args = {"topic": x}
    if d:
        args["depth"] = d
    return _fill(f, X=x, V=v), args


CAL_VERBS = ["takvime ekle", "takvimime koy", "takvime yaz", "ajandama yaz", "ajandaya ekle", "randevu oluştur",
             "etkinlik oluştur", "programıma ekle", "takvime kaydet", "planla", "takvime işle", "etkinlik ekle"]
CAL_FRAMES = ["{T} {X} {V}", "{X} için {T} {V}", "{V}: {X}, {T}", "{X}, {T}, {V}", "{T} {X} var, {V}",
              "{T} {X} etkinliği, {V}", "{X} {T}, {V}", "{V}, {T} {X}"]


def g_calendar_add(rng, sp, hf, hv):
    t = random_time(rng, allow_offset=False)
    x = pick(rng, EVENT_TITLES, sp, lambda r: random_phrase(r, 1, 3))
    v = _verb(rng, sp, CAL_VERBS, hv)
    args = {"title": x, "start": t.when}
    s = _fill(_frame(rng, sp, CAL_FRAMES, hf), T=t.tr, X=x, V=v)
    r = rng.random()
    if r < 0.12:
        n = rng.choice([15, 30, 45, 90, 120]); s += f", {n} dakika sürecek"; args["minutes"] = n
    elif r < 0.2:
        p = pick(rng, PLACES, sp, lambda q: random_phrase(q, 1, 2, True), 0.5); s += f", yer {p}"; args["place"] = p
    return s, args


DAYS = [("bugün", "d+0"), ("yarın", "d+1"), ("öbür gün", "d+2"), ("bu gün", "d+0"), ("ertesi gün", "d+2")]
CL_PRED = ["takvimde ne var", "programım ne", "neler var", "boş muyum", "işim var mı", "etkinliklerimi göster",
           "ne yapacağım", "randevum var mı", "takvimime bak", "planım ne", "programımda ne var", "meşgul müyüm"]


def g_calendar_list(rng, sp, hf, hv):
    if rng.random() < 0.25:
        n = rng.randint(3, 9); tr, d = f"{n} gün sonra", f"d+{n}"
    else:
        tr, d = rng.choice(DAYS)
    p = G.choose(rng, CL_PRED, sp, hf or hv)
    return (f"{tr} {p}" if rng.random() < 0.8 else f"{p}, {tr}"), {"day": d}


TIMER_VERBS = ["zamanlayıcı kur", "geri sayım başlat", "sayaç başlat", "say", "süre tut", "zamanlayıcı başlat",
               "sayaç kur", "geri sayım yap"]
TIMER_FRAMES = ["{M} dakikalık {V}", "{M} dakika {V}", "{L} için {M} dakika {V}", "{V}: {M} dakika",
                "{M} dakikalık {L} {V}", "{L} için {M} dakikalık {V}", "{V}, {M} dakika", "{M} dk {V}"]


def g_timer(rng, sp, hf, hv):
    m = rng.choice([1, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 45, 60, 90, 120])
    f = _frame(rng, sp, TIMER_FRAMES, hf)
    v = _verb(rng, sp, TIMER_VERBS, hv)
    args = {"minutes": m}
    if "{L}" in f:
        lb = pick(rng, TIMER_LABELS, sp, lambda r: random_phrase(r, 1, 2))
        args["label"] = lb
        return _fill(f, M=str(m), L=lb, V=v), args
    return _fill(f, M=str(m), V=v), args


MEM_NOTE = ["bunu hatırla", "şunu aklında tut", "aklında olsun", "kalıcı hafızaya yaz", "hafızana kaydet",
            "bunu unutma", "şunu öğren", "bunu bil"]
MEM_PREF = ["tercihim şu", "benim için kural", "bundan sonra şöyle yap", "bir tercihim var", "kuralım şu",
            "her zaman şöyle olsun", "alışkanlığım şu", "tercihimi kaydet"]
MEM_SKILL = ["bu adımları beceri olarak kaydet", "beceri olarak kaydet", "bunu beceri olarak öğren", "yeni beceri"]


def g_memory(rng, sp, hf, hv):
    x = pick(rng, MEMORY_TEXTS, sp, lambda r: random_phrase(r, 2, 7))
    r = rng.random()
    kind, pool = ("note", MEM_NOTE) if r < 0.5 else ("preference", MEM_PREF) if r < 0.85 else ("skill", MEM_SKILL)
    return f"{G.choose(rng, pool, sp, hf or hv)}: {x}", {"kind": kind, "text": x}


HELP_PHR = ["neler yapabilirsin", "ne yapabiliyorsun", "yeteneklerin neler", "nasıl kullanılırsın", "komutların neler",
            "ne işe yararsın", "hangi işleri yaparsın", "yardım", "bana neler yaparsın", "neler biliyorsun",
            "kullanım kılavuzu", "ne yapabildiğini göster", "becerilerin neler", "yardım menüsü", "nasıl yardımcı olabilirsin",
            "seninle neler yapabilirim"]


def g_help(rng, sp, hf, hv):
    return G.choose(rng, HELP_PHR, sp, hf or hv), {}


ASK_FRAMES = ["{X} nedir", "{X} nasıl yapılır", "{X} ne demek", "{X} neden önemli", "{X} hakkında bilgi ver",
              "{X} nasıl çalışır", "{X} ne işe yarar", "{X} nerede bulunur", "{X} kimdir", "{X} nasıl öğrenilir",
              "{X} ile ilgili ne biliyorsun", "{X} iyi mi kötü mü"]
ASK_PRE = ["", "", "", "soru: ", "bana anlat: ", "merak ediyorum, ", "açıkla: ", "sorum şu: ", "bir sorum var: "]


def g_ask(rng, sp, hf, hv):
    x = pick(rng, ["kara delik", "fotosentez", "yapay zeka", "Osmanlı İmparatorluğu", "kredi kartı faizi", "bulut depolama",
                   "elektrikli araba", "uyku düzeni", "Python dili", "kuantum bilgisayar", "Mars", "enflasyon",
                   "ekmek mayası", "güneş paneli", "deprem", "bağışıklık sistemi"], sp, lambda r: random_phrase(r, 1, 4))
    q = _fill(_frame(rng, sp, ASK_FRAMES, hf or hv), X=x)
    pre = rng.choice(ASK_PRE)
    return pre + q + ("?" if not pre and rng.random() < 0.5 else ""), {"question": q}


def g_calc(rng, sp, hf, hv):
    def n(): return rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 25, 40, 64, 75, 100, 150, 250, 1250, 3600])
    a, b = n(), n()
    forms = [("{a} artı {b} kaç", "+"), ("{a} eksi {b} kaç", "-"), ("{a} çarpı {b} kaç", "*"), ("{a} bölü {b} kaç", "/"),
             ("{a} ile {b} toplamı kaç", "+"), ("{a} ile {b} farkı kaç", "-"), ("{a} ile {b} çarpımı kaç", "*"),
             ("{a} kere {b} kaç eder", "*"), ("hesapla: {a}+{b}", "+"), ("hesapla: {a}*{b}", "*"),
             ("{a} x {b} kaç", "*"), ("{a} bölü {b} ne eder", "/")]
    f, op = G.choose(rng, forms, sp, hf or hv)
    return f.format(a=a, b=b), {"expr": f"{a}{op}{b}"}


LANG = [("İngilizce", "en"), ("Almanca", "de"), ("Fransızca", "fr"), ("İspanyolca", "es"), ("Arapça", "ar"),
        ("Rusça", "ru"), ("İtalyanca", "it"), ("Türkçe", "tr")]
TR_FRAMES = ["{LD} çevir: {X}", "{LD} tercüme et: {X}", "şunu {L} yap: {X}", "{L} karşılığı ne: {X}",
             "{L} olarak yaz: {X}", "{L} nasıl söylenir: {X}", "bunu {LD} çevirir misin: {X}", "{L} çevirisi: {X}"]


def g_translate(rng, sp, hf, hv):
    x = pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 2, 8))
    ln, lc = rng.choice(LANG)
    return _fill(_frame(rng, sp, TR_FRAMES, hf or hv), L=ln, LD=G.dative(ln), X=x), {"text": x, "to": lc}


IMGA = [("ekteki fotoğrafı oku", None), ("bu ekran görüntüsünde ne yazıyor", None), ("fotoğraftaki yazıyı çıkar", None),
        ("resimdeki yazıyı oku", None), ("ss'i oku", None), ("görseldeki metni çıkar", None),
        ("ekran görüntüsündeki sorunu çöz", "sorunu çöz"), ("fotoğraftaki hatayı çöz", "hatayı çöz")]
IMGA_Q = ["resmi incele: {Q}", "ekran görüntüsüne bak ve şunu cevapla: {Q}", "fotoğrafa bakıp söyle: {Q}",
          "görsele bak: {Q}", "ss'e bakar mısın: {Q}", "resimde şunu bul: {Q}"]


def g_image_analyze(rng, sp, hf, hv):
    if rng.random() < 0.4:
        s, q = G.choose(rng, IMGA, sp, hf or hv)
        return s, ({"uri": ATTACHED, "question": q} if q else {"uri": ATTACHED})
    q = pick(rng, QUESTIONS, sp, lambda r: random_phrase(r, 2, 6))
    return _fill(G.choose(rng, IMGA_Q, sp, hf or hv), Q=q), {"uri": ATTACHED, "question": q}


IMGG_VERBS = ["resmi çiz", "görseli oluştur", "resmi yap", "görseli üret", "resmi oluştur", "çizimi yap",
              "illüstrasyonu hazırla", "resmini çiz"]
IMGG_PRE = ["resim oluştur: {X}", "görsel üret: {X}", "bana bir resim çiz: {X}", "çiz: {X}"]


def g_image_generate(rng, sp, hf, hv):
    x = pick(rng, IMAGE_PROMPTS, sp, lambda r: random_phrase(r, 2, 6))
    args = {"prompt": x}
    if rng.random() < 0.35:
        return _fill(G.choose(rng, IMGG_PRE, sp, hf or hv), X=x), args
    v = G.choose(rng, IMGG_VERBS, sp, hv)
    size = rng.random()
    pre = ""
    if size < 0.12:
        pre = "dikey bir "; args["size"] = "portrait"
    elif size < 0.24:
        pre = "yatay bir "; args["size"] = "landscape"
    return f"{pre}{x} {v}".strip(), args


FILE_SUBJ = [("ekteki dosyayı", "ekteki dosyanın"), ("bu dosyayı", "bu dosyanın"), ("eklediğim belgeyi", "eklediğim belgenin"),
             ("ekteki pdf'i", "ekteki pdf'in"), ("bu belgeyi", "bu belgenin"), ("eki", "ekin"),
             ("gönderdiğim dosyayı", "gönderdiğim dosyanın"), ("ekli metni", "ekli metnin")]


def g_file(rng, sp, hf, hv):
    acc, gen = G.choose(rng, FILE_SUBJ, sp, hf)
    ln = rng.choice([("", None), ("kısaca ", "short"), ("uzun uzun ", "long"), ("ayrıntılı ", "long"), ("orta uzunlukta ", "medium")])
    verb = G.choose(rng, ["özetle", "özetler misin", "özetini çıkar", "özet geç"], sp, hv)
    args = {"uri": ATTACHED}
    if ln[1]:
        args["length"] = ln[1]
    if verb == "özetini çıkar":
        return f"{gen} {ln[0]}özetini çıkar", args
    if verb == "özet geç":
        return f"{gen} {ln[0]}özetini geç", args
    return f"{acc} {ln[0]}{verb}", args


REPO_FRAMES = ["{R} reposunu {V}", "{R} deposunu {V}", "{R} projesini {V}", "{V}: {R}", "{R} için {V}",
               "{R} projesinin durumunu {V}", "{R} reposunun son halini {V}"]
REPO_VERBS = ["denetle", "kontrol et", "incele", "gözden geçir", "bir bak", "analiz et", "tara", "değerlendir", "özetle"]


def g_repo(rng, sp, hf, hv):
    r = pick(rng, REPOS, sp, random_repo, 0.85)
    args = {"repo": r}
    if rng.random() < 0.3:
        sc, w = rng.choice([("ci", "CI durumunu"), ("ci", "derleme durumunu"), ("security", "güvenliğini")])
        args["scope"] = sc
        return f"{r} {w} {G.choose(rng, REPO_VERBS, sp, hv)}", args
    return _fill(G.choose(rng, REPO_FRAMES, sp, hf), R=r, V=G.choose(rng, REPO_VERBS, sp, hv)), args


def g_card(rng, sp, hf, hv):
    kind = rng.choice(list(CARD_KIND_WORDS)); w = rng.choice(CARD_KIND_WORDS[kind])
    t = pick(rng, CARD_TITLES, sp, lambda r: random_phrase(r, 1, 3, True))
    f = G.choose(rng, ["{W} kartı ekle: {T}", "ana ekrana {W} kartı koy: {T}", "{T} için {W} kartı yap",
                       "'{T}' başlıklı bir {W} kartı ekle", "yeni {W} kartı: {T}", "{T} başlıklı {W} kartı oluştur",
                       "ekrana {W} kartı ekle, adı {T}", "{W} kartı istiyorum: {T}",
                       "{T} için bir {W} kartı hazırla", "ana ekranıma {T} gösteren bir {W} kartı koy",
                       "{T} için {W} kartı tasarla", "{T} gösteren {W} kartı ekle"], sp, hf or hv)
    return _fill(f, W=w, T=t), {"kind": kind, "title": t}


def g_discover(rng, sp, hf, hv):
    if rng.random() < 0.4:
        x = pick(rng, NEEDS, sp, lambda r: random_phrase(r, 1, 3))
        f = G.choose(rng, ["{N} için ücretsiz yapay zeka bul", "{N} yapabilen bedava bir model öner",
                           "{N} için ücretsiz bir model ara", "{N} işine yarayan bedava yapay zeka bul"], sp, hf or hv)
        return _fill(f, N=x), {"need": x}
    return G.choose(rng, ["yeni ücretsiz yapay zeka bul", "başka ücretsiz model var mı araştır", "beyne eklenebilecek yeni sağlayıcı ara",
                          "kotası dolmayan yeni bir API bul", "bedava yapay zeka sağlayıcısı bul", "yeni bir ücretsiz model öner",
                          "ücretsiz API ara", "yeni sağlayıcı bul"], sp, hf or hv), {}


def g_health(rng, sp, hf, hv):
    s, upd = G.choose(rng, [("telefonun sağlığını kontrol et", False), ("güncellemeden önce telefonu kontrol et", True),
                            ("pil ve depolama durumu nasıl", False), ("güncelleme yapsam sorun olur mu", True),
                            ("telefon sağlıklı mı", False), ("sistem güncellemesine hazır mıyım", True),
                            ("telefonumun durumu nasıl", False), ("güncellemeye hazır mıyım", True)], sp, hf or hv)
    return s, ({"purpose": "update"} if upd else {})


COINS = ["bitcoin", "ethereum", "solana", "dogecoin", "XRP", "cardano", "avalanche", "BNB", "tron", "polkadot",
         "litecoin", "chainlink", "shiba", "toncoin", "pepe", "sui"]
CRYPTO_ALL = ["bugün kripto piyasası nasıl", "coin borsası ne durumda", "kripto paralara bak", "kripto piyasasına göz at",
              "coinler ne alemde", "piyasa nasıl, kripto tarafı", "bugün coinler nasıl gidiyor", "kripto durumu ne",
              "coin piyasasını göster", "kriptolar yükselmiş mi", "kripto borsasında durum nasıl", "coin fiyatlarına bak"]
CRYPTO_ONE = ["{C} kaç {P}", "{C} fiyatı ne", "{C} ne durumda", "{C} kaç para", "{C} bugün nasıl", "{C} yükseldi mi",
              "{C} fiyatına bak", "{C} ne kadar oldu"]
CUR_WORDS = [("lira", "try"), ("TL", "try"), ("dolar", "usd"), ("euro", "eur")]


def g_crypto(rng, sp, hf, hv):
    if rng.random() < 0.4:
        s = G.choose(rng, CRYPTO_ALL, sp, hf or hv)
        if rng.random() < 0.3:
            w, c = rng.choice(CUR_WORDS)
            return f"{s}, {w} olarak", {"currency": c}
        return s, {}
    # Rastgele kelime kullanılmaz: "domates kaç TL" kripto sorusu sanılmasın.
    coin = pick(rng, COINS, sp)
    f = G.choose(rng, CRYPTO_ONE, sp, hf or hv)
    w, c = rng.choice(CUR_WORDS)
    args = {"coin": coin}
    if "{P}" in f:
        if c != "try" or rng.random() < 0.5:
            args["currency"] = c
        return _fill(f, C=coin, P=w), args
    return _fill(f, C=coin), args


SKILL_NAMES = ["sabah rutini", "akşam kontrolü", "kripto özeti", "haftalık plan", "iş başlangıcı", "spor takibi",
               "okuma listesi", "gün sonu raporu", "proje durumu", "hafta sonu planı", "fatura kontrolü", "kod çalışması"]
SKILL_GOALS = ["her sabah takvimimi ve kripto piyasasını özetle", "github'da yapay zeka ajanları için iyi repolar bul ve not al",
               "haftalık harcamalarımı notlardan derle", "bana her akşam yarının planını çıkar", "yeni çıkan Android güncellemelerini takip et",
               "okuduğum kitapları not alıp haftalık özetle", "anahtarlı yapay zekalar için kullanılabilir bir çözüm bul",
               "işe yarar ücretsiz araçlar bulup listele"]
ROLES = ["kaynakları doğrulayan titiz araştırmacı", "kod örneklerini inceleyen yazılımcı", "kısa ve net özet çıkaran editör",
         "fiyat ve piyasa verisini yorumlayan analist", "Türkçe metinleri düzelten yazım uzmanı", "görsel fikir üreten tasarımcı"]
AGENT_TR = [("Sophia", "sophia", "'ya"), ("Jack", "jack", "'e"), ("Emma", "emma", "'ya"), ("Chloe", "chloe", "'ye"),
            ("Luna", "luna", "'ya"), ("Mike", "mike", "'a"), ("David", "david", "'e"), ("Maya", "maya", "'ya"), ("Grace", "grace", "'e")]


def g_skill_run(rng, sp, hf, hv):
    x = pick(rng, SKILL_NAMES, sp, lambda r: random_phrase(r, 1, 3))
    v = G.choose(rng, ["çalıştır", "kullan", "uygula", "başlat", "yap"], sp, hv)
    if rng.random() < 0.35:
        inp = pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 1, 5))
        return f"{x} becerisini {v}: {inp}", {"name": x, "input": inp}
    return f"{x} becerisini {v}", {"name": x}


def g_skill_list(rng, sp, hf, hv):
    return G.choose(rng, ["becerilerimi göster", "becerilerin neler", "hangi becerilerin var", "becerileri listele",
                          "araçlarını göster", "dış ajanları göster", "becerilerini göster", "becerilerin ne"], sp, hf or hv), {}


def g_skill_learn(rng, sp, hf, hv):
    goal = pick(rng, SKILL_GOALS, sp, lambda r: random_phrase(r, 3, 8))
    pre = G.choose(rng, ["beceri öğren", "yeni beceri öğren", "şunu yapmayı öğren", "yeni beceri"], sp, hf or hv)
    return f"{pre}: {goal}", {"goal": goal}


def g_skill_record(rng, sp, hf, hv):
    x = pick(rng, SKILL_NAMES, sp, lambda r: random_phrase(r, 1, 3))
    if rng.random() < 0.6:
        n = rng.randint(1, 8)
        w = G.choose(rng, ["işi", "adımı", "komutu"], sp, hv)
        return f"son {n} {w} beceri olarak kaydet: {x}", {"name": x, "count": n}
    pre = G.choose(rng, ["son yaptıklarımı", "bunları", "bunu"], sp, hf)
    return f"{pre} beceri olarak kaydet: {x}", {"name": x}


def g_agent_hire(rng, sp, hf, hv):
    role = pick(rng, ROLES, sp, lambda r: random_phrase(r, 2, 6))
    v = G.choose(rng, ["yardımcı ajan ekle", "yardımcı al", "dış ajan ekle", "yardımcı ajan al"], sp, hf or hv)
    if rng.random() < 0.7:
        name, aid, suf = rng.choice(AGENT_TR)
        return f"{name}{suf} {v}: {role}", {"role": role, "helps": aid}
    return f"{v}: {role}", {"role": role}


MAIL_CHECK = ["yeni maillerime bak", "e-postalarımı kontrol et", "gelen kutuma bak", "yeni mail var mı", "maillerimi oku",
              "e-postalarım neler", "gelen kutusunu göster", "postalarıma bak", "mail geldi mi", "e-posta kutumu kontrol et"]
MAIL_TO = ["ali@ornek.com", "ayse.yilmaz@gmail.com", "info@sirket.com.tr", "destek@uygulama.io", "mehmet@okul.edu.tr"]
MAIL_SUBJ = ["Yarınki toplantı", "Fatura hakkında", "Proje güncellemesi", "Teşekkürler", "Randevu talebi", "Haftalık rapor"]


def g_mail_check(rng, sp, hf, hv):
    s = G.choose(rng, MAIL_CHECK, sp, hf or hv)
    r = rng.random()
    if r < 0.25:
        n = rng.choice([3, 5, 10, 20])
        return f"son {n} maili göster", {"count": n}
    if r < 0.4:
        who = rng.choice(NAMES)
        return f"{who}'dan mail geldi mi", {"query": who}
    return s, {}


def g_mail_send(rng, sp, hf, hv):
    to = rng.choice(MAIL_TO)
    subj = pick(rng, MAIL_SUBJ, sp, lambda r: random_phrase(r, 1, 4, True))
    body = pick(rng, NOTE_TEXTS, sp, lambda r: random_phrase(r, 3, 10))
    v = G.choose(rng, ["mail yaz", "e-posta gönder", "mail at", "e-posta hazırla"], sp, hf or hv)
    return f"{to} adresine {v}: konu: {subj}; {body}", {"to": to, "subject": subj, "body": body}


GRAMMAR = {
    "reminder.set": g_reminder, "reminder.list": g_reminder_list, "note.write": g_note, "note.search": g_note_search,
    "research.start": g_research, "repo.audit": g_repo, "calendar.add": g_calendar_add, "calendar.list": g_calendar_list,
    "file.summarize": g_file, "image.analyze": g_image_analyze, "image.generate": g_image_generate, "card.propose": g_card,
    "timer.start": g_timer, "provider.discover": g_discover, "health.check": g_health, "memory.save": g_memory,
    "ask": g_ask, "calc.eval": g_calc, "text.translate": g_translate, "help": g_help, "market.crypto": g_crypto,
    "skill.run": g_skill_run, "skill.list": g_skill_list, "skill.learn": g_skill_learn, "skill.record": g_skill_record,
    "agent.hire": g_agent_hire, "mail.check": g_mail_check, "mail.send": g_mail_send,
}
GRAMMAR_FRAC = 0.75


AGENT_NAMES = ["Mike", "Sophia", "Emma", "Luna", "David", "Jack", "Chloe", "Grace", "Maya", "Orion"]


def grammar_example(rng: random.Random, cmd: str, split: str) -> tuple[str, dict]:
    hf = hv = False
    if split == "test_ood":
        r = rng.random()
        hf, hv = (True, False) if r < 0.4 else (False, True) if r < 0.8 else (True, True)
    tr, args = GRAMMAR[cmd](rng, split, hf, hv)
    # Ajana seslenme ("Luna, ..." / "Luna ...") hiçbir argümanın parçası değildir (Tur 6 kök sebep).
    if rng.random() < 0.12:
        tr = rng.choice(AGENT_NAMES) + rng.choice([" ", ", "]) + tr
    return tr, args
