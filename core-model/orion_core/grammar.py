"""Dilbilgisi motoru: Türkçe isteği parçalardan kurarak çok çeşitli eğitim cümlesi üretir.

Neden: Tur 1-2'de model yalnızca 98 sabit cümle kalıbıyla eğitildi ve onları ezberledi
(görülmemiş kalıp testinde cümlelerin %89,5'inde eğitimde HİÇ geçmeyen kelimeler vardı).
Burada her komut için eş anlamlı fiil havuzu × söz dizimi çerçevesi × istek kipi
(emir, "-ır mısın", "-abilir misin") × nezaket birleştirilir; binlerce farklı yapı çıkar.

Bölme (sızıntı olmasın diye her havuz sabit kuralla ayrılır):
  - fiiller ve çerçeveler: her 4. öğe (i % 4 == 3) yalnızca test_ood'da kullanılır,
  - değerler (not metni, konu...): datagen._split ile (son %25 test_iid/test_ood'a).
test_ood örneğinde en az bir parça (çerçeve ya da fiil) eğitimde hiç görülmemiştir.
"""
from __future__ import annotations

import random
from typing import Callable

VOWELS = "aeıioöuü"
BACK = "aıou"


def _last_vowel(w: str) -> str:
    for c in reversed(w.lower()):
        if c in VOWELS:
            return c
    return "e"


def h2(w: str) -> str:
    return "a" if _last_vowel(w) in BACK else "e"


def h4(w: str) -> str:
    v = _last_vowel(w)
    return {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}[v]


def _syllables(w: str) -> int:
    return sum(c in VOWELS for c in w.lower())


# Tek heceli fiillerin geniş zamanı (düzensizler -ır/-ir; çoğu -ar/-er)
_AOR_IRREG = {"al": "alır", "gel": "gelir", "ver": "verir", "bul": "bulur", "ol": "olur", "gör": "görür",
              "kal": "kalır", "bil": "bilir", "dur": "durur", "san": "sanır", "var": "varır", "vur": "vurur",
              "öl": "ölür", "an": "anar", "et": "eder", "git": "gider", "de": "der", "ye": "yer"}


def aorist(verb: str) -> str:
    """Fiil öbeğinin son kelimesini geniş zamana çeker: hatırlat → hatırlatır, not et → not eder."""
    words = verb.split(" ")
    w = words[-1]
    if w in _AOR_IRREG:
        a = _AOR_IRREG[w]
    elif w.endswith("et") and _syllables(w) > 1 and w in ("kaydet", "terket", "yardımet"):
        a = w[:-1] + "der"
    elif w[-1] in VOWELS:
        a = w + "r"
    elif _syllables(w) == 1:
        a = w + h2(w) + "r"
    else:
        a = w + h4(w) + "r"
    words[-1] = a
    return " ".join(words)


def ability(verb: str) -> str:
    """hatırlat → hatırlatabilir, not et → not edebilir, ara → arayabilir."""
    words = verb.split(" ")
    w = words[-1]
    stem = {"et": "ed", "git": "gid", "kaydet": "kayded", "de": "diy", "ye": "yiy"}.get(w, w)
    if stem[-1] in VOWELS:
        stem += "y"
    words[-1] = stem + h2(stem) + "bilir"
    return " ".join(words)


def q_particle(word: str) -> str:
    v = h4(word)
    return f"m{v}s{v}n"


def moods(verb: str) -> list[str]:
    """Aynı isteğin söyleniş biçimleri."""
    a = aorist(verb)
    b = ability(verb)
    return [verb, f"{a} {q_particle(a)}", f"{b} {q_particle(b)}"]


def dative(word: str) -> str:
    """İngilizce → İngilizceye, Almanca → Almancaya."""
    if word[-1].lower() in VOWELS:
        return word + "y" + h2(word)
    return word + h2(word)


def inf_accusative(inf: str) -> str | None:
    """'ilaç içmek' → 'ilaç içmeyi' (mastarın belirtme hâli); mastar değilse None."""
    if inf.endswith("mek"):
        return inf[:-1] + "yi"
    if inf.endswith("mak"):
        return inf[:-1] + "yı"
    return None


# --------------------------------------------------------------------------- bölme

def split_pool(items: list, split: str) -> list:
    """Fiil/çerçeve havuzu: her 4. öğe yalnızca test_ood içindir."""
    train = [x for i, x in enumerate(items) if i % 4 != 3]
    held = [x for i, x in enumerate(items) if i % 4 == 3] or train
    return held if split == "held" else train


def choose(rng: random.Random, items: list, split: str, held_ok: bool) -> object:
    """test_ood'da held_ok=True olan parça görülmemiş havuzdan seçilir."""
    if split == "test_ood" and held_ok:
        return rng.choice(split_pool(items, "held"))
    return rng.choice(split_pool(items, "train"))


Spec = Callable[[random.Random, str, bool, bool], tuple[str, dict]]
