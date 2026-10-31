"""Hafıza notlu diyaloglar: uygulamanın katmanlı hafızasıyla sohbet beyni arasındaki köprü.

BİÇİM (uygulama/Kotlin tarafıyla AYNI; değiştirmeden önce iki tarafı birlikte güncelle):
    Hafıza: <not1>; <not2>; <not3>\\n
    Kullanıcı: <soru>\\nOrion: <cevap>            (eğitim belgesi; çok turluysa turlar alt alta)
    Kullanıcı: <soru>\\nOrion:                     (çıkarım istemi: chat.prompt_for(soru, memory="<not1>; <not2>"))
Kurallar (memory_line / validate_line denetler):
- "Hafıza: " ön ekiyle BİRLİKTE satır en çok 160 karakter (satır sonu hariç); en çok 3 not, ayraç "; "
- her not en çok 70 karakter, yarıda kesilmez (sığmayan not bütünüyle düşer), içinde ";" ve satır sonu yok,
  sonunda "." yok; özel ad değilse küçük harfle başlar
- notlar ikinci tekil kişiyle sahibine dair gerçeklerdir: "kızının adı Ela", "34 yaşındasın",
  "Konya'da yaşıyorsun", "kahveyi şekersiz içiyorsun"
- hafıza yoksa satır HİÇ yazılmaz (eski biçim aynen: "Kullanıcı: …\\nOrion:")
Belirteçleme: belge <bos> + bpe.encode(tüm metin) + <eos>; istem <bos> + bpe.encode(istem) — Python (chat.generate)
ve Kotlin (ChatBrain) aynı metni aynı tokenizer'la kodlar; "Hafıza:" satırı için özel belirteç YOKTUR.

Davranış: Orion ilgili notu doğal biçimde KULLANIR, ilgisiz notları anmaz, notta olmayan bir şeyi UYDURMAZ (sorulursa
bilmediğini söyler ve sorar). Bütün adlar/tarihler kurgusaldır; gerçek kullanıcı verisi yoktur. Bilişsel sınavdaki
adlar (Defne, Fıstık, Trabzon…) kullanılmaz (test denetler).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

PREFIX = "Hafıza: "
SEP = "; "
MAX_LINE = 160
MAX_NOTES = 3
MAX_NOTE = 70


def _cap(s: str) -> str:
    c = s[:1]
    return ("İ" if c == "i" else "I" if c == "ı" else c.upper()) + s[1:]


def valid_note(n: str) -> bool:
    return (0 < len(n) <= MAX_NOTE and ";" not in n and "\n" not in n and not n.endswith(".")
            and n == n.strip())


def memory_line(notes: list[str]) -> str:
    """Notlardan "Hafıza: …" satırı (satır sonu YOK). Sığmayan not bütünüyle düşer (sondan); hiç not kalmazsa ""."""
    notes = [n for n in notes if valid_note(n)][:MAX_NOTES]
    while notes and len(PREFIX + SEP.join(notes)) > MAX_LINE:
        notes = notes[:-1]
    return PREFIX + SEP.join(notes) if notes else ""


def validate_line(line: str, proper_starts: set[str] | None = None) -> list[str]:
    """Kural ihlalleri (boş liste = geçerli). proper_starts: büyük harfle başlayabilecek özel adlar."""
    errs = []
    if not line.startswith(PREFIX):
        errs.append("önek")
    if len(line) > MAX_LINE:
        errs.append(f"uzun: {len(line)}")
    if "\n" in line:
        errs.append("satır sonu")
    notes = line[len(PREFIX):].split(SEP)
    if not 1 <= len(notes) <= MAX_NOTES:
        errs.append(f"not sayısı: {len(notes)}")
    for n in notes:
        if not valid_note(n):
            errs.append(f"geçersiz not: {n!r}")
        first = n.split(" ", 1)[0]
        if n[:1].isupper() and not (proper_starts and re.split(r"['’]", first)[0] in proper_starts):
            errs.append(f"büyük harf: {n!r}")
    return errs


# ------------------------------------------------------------------ kurgusal değerler
GIRL_NAMES = ["Ada", "Nehir", "Ela", "Duru", "Asya", "Mira", "Lina"]
BOY_NAMES = ["Yağız", "Eymen", "Aras", "Kuzey", "Toprak", "Efe", "Alp"]
KID_NAMES = GIRL_NAMES + BOY_NAMES
PARTNER_NAMES = ["Selim", "Burak", "Emre", "Gökhan", "Cem", "Melis", "Cansu", "İpek", "Seda", "Damla"]
PET_NAMES = ["Tarçın", "Zeytin", "Boncuk", "Paşa", "Limon", "Karamel", "Duman", "Mırmır", "Badem"]
CITIES = [("İzmir", "İzmir'de"), ("Bursa", "Bursa'da"), ("Antalya", "Antalya'da"), ("Konya", "Konya'da"),
          ("Samsun", "Samsun'da"), ("Gaziantep", "Gaziantep'te"), ("Mersin", "Mersin'de"), ("Ankara", "Ankara'da"),
          ("Edirne", "Edirne'de"), ("Malatya", "Malatya'da")]
MONTHS = [("Ocak", "'ta"), ("Şubat", "'ta"), ("Mart", "'ta"), ("Nisan", "'da"), ("Mayıs", "'ta"), ("Haziran", "'da"),
          ("Temmuz", "'da"), ("Ağustos", "'ta"), ("Eylül", "'de"), ("Ekim", "'de"), ("Kasım", "'ta"), ("Aralık", "'ta")]
# Büyük harfle başlayabilen notların ilk sözcükleri (özel adlar)
PROPER = {c for c, _ in CITIES} | {"İspanyolca", "İngilizce", "Almanca"}


@dataclass
class Fact:
    kind: str
    note: str
    values: set[str] = field(default_factory=set)       # notun özgül değerleri (ad, tarih, şehir, yaş)
    use: list[tuple[str, str]] = field(default_factory=list)     # notu KULLANAN (soru, cevap)
    recall: list[tuple[str, str]] = field(default_factory=list)  # notu doğrudan hatırlatan (soru, cevap)


# notta yoksa: (soru, "bilmiyorum" cevapları) — tür başına
ABSENT = {
    "child": ("Kızımın adını biliyor musun?", "Bilmiyorum; hafızamda kızının adı yok."),
    "partner": ("Eşimin adı neydi, biliyor musun?", "Bilmiyorum; hafızamda eşinin adı yok."),
    "pet": ("Kedimin adını biliyor musun?", "Bilmiyorum; hafızamda kedinle ilgili bir not yok."),
    "drink": ("Kahveyi nasıl içtiğimi biliyor musun?", "Bilmiyorum; hafızamda bununla ilgili bir not yok."),
    "weekly": ("Hafta içi düzenli bir aktivitem var mıydı?", "Hafızamda buna dair bir not yok."),
    "birthday": ("Babamın yaş günü hangi tarihteydi?", "Bunu bilmiyorum; hafızamda babanın doğum günü yok."),
    "diet": ("Benim bir alerjim var mıydı?", "Hafızamda alerjinle ilgili bir not yok; tahmin etmek de istemem."),
    "wake": ("Sabahları kaçta kalktığımı biliyor musun?", "Bilmiyorum; hafızamda kalkış saatinle ilgili bir not yok."),
    "goal": ("Şu sıralar ne öğrendiğimi biliyor musun?", "Bilmiyorum; hafızamda buna dair bir not yok."),
    "city": ("Hangi şehirde yaşadığımı biliyor musun?", "Bilmiyorum; hafızamda yaşadığın şehir yok."),
    "age": ("Kaç yaşında olduğumu biliyor musun?", "Bilmiyorum; hafızamda yaşın yok."),
}
ABSENT_ASK = ["Söylersen not alayım mı?", "Söyler misin, hafızama ekleyelim mi?", "Anlatırsan not alayım mı?",
              "Tahmin etmek yerine sana sorayım: söyler misin?"]
RECALL_TAIL = ["", "Hafızamda öyle duruyor.", "Yanlışsa düzeltirsin."]


def _child(rng):
    girl = rng.random() < 0.5
    poss, my, myg = ("kızının", "Kızım", "Kızımın") if girl else ("oğlunun", "Oğlum", "Oğlumun")
    n = rng.choice(GIRL_NAMES if girl else BOY_NAMES)
    return Fact("child", f"{poss} adı {n}", {n},
                use=[(f"{my} için doğum günü kartına ne yazayım?",
                      f"\"Sevgili {n}, iyi ki doğdun; seninle her gün daha güzel\" diye başlayabilirsin. Kısa ve içten olsun."),
                     (f"{my}la bu hafta sonu ne yapabiliriz?",
                      f"{n} ile parkta bisiklete binebilir ya da evde birlikte kek yapabilirsiniz; ikisi de keyifli."),
                     (f"{my} bu akşam uyumak istemiyor, ne yapsam?",
                      f"{n} için kısa bir masal ve loş bir ışık iyi gelebilir; ekranları da yarım saat önce kapatın.")],
                recall=[(f"{myg} adını bir de sen söyle, doğru mu kaydetmişsin bakayım", f"{n}.")])


def _partner(rng):
    n = rng.choice(PARTNER_NAMES)
    return Fact("partner", f"eşinin adı {n}", {n},
                use=[("Eşime bu akşam güzel bir mesaj yazmak istiyorum, ne yazayım?",
                      f"\"{n}, bugün aklım hep sendeydi; akşamı iple çekiyorum.\" gibi kısa ve sıcak bir şey olabilir."),
                     ("Eşimle bu akşam ne yapsak?",
                      f"{n} ile evde güzel bir yemek ve sonra kısa bir yürüyüş nasıl? Sakin ama keyifli.")],
                recall=[("Eşimin adını doğru kaydetmiş misin?", f"{n} olarak kayıtlı.")])


def _pet(rng):
    cat = rng.random() < 0.5
    poss, to, my = ("kedinin", "Kedime", "Kedimin") if cat else ("köpeğinin", "Köpeğime", "Köpeğimin")
    p = rng.choice(PET_NAMES)
    toy = (f"{p} için tüylü bir olta oyuncağı ya da içinde kedi nanesi olan küçük bir top iyi olur."
           if cat else f"{p} için sağlam bir çiğneme oyuncağı ya da fırlatma topu iyi olur; yürüyüşte de oynarsınız.")
    return Fact("pet", f"{poss} adı {p}", {p},
                use=[(f"{to} bir oyuncak almak istiyorum, ne önerirsin?", toy)],
                recall=[(f"{my} adını sen söyle bakalım", f"{p}! Unutur muyum hiç.")])


DRINKS = [("kahveyi şekersiz içiyorsun", "Şekersiz bir filtre kahve tam sana göre"),
          ("kahveyi sütlü seviyorsun", "Sütlü bir kahve tam sana göre"),
          ("çayı açık seviyorsun", "Açık demlenmiş bir çay tam sana göre"),
          ("bitki çayını seviyorsun", "Bir bardak ıhlamur ya da adaçayı tam sana göre")]


def _drink(rng):
    note, piece = rng.choice(DRINKS)
    tail = rng.choice(["yanına birkaç ceviz de iyi gider.", "beş dakika pencere kenarında içersen mola tam olur.",
                       "telefonu da biraz bırak, molanın hakkını ver."])
    return Fact("drink", note, set(),
                use=[(q, f"{piece}; {tail}") for q in ["Kısa bir mola vereceğim, ne içsem?",
                                                       "Çalışırken yanıma ne alsam, bir içecek öner",
                                                       "Akşamüstü canım bir şey içmek istiyor, ne önerirsin?"]])


DAYS = [("Pazartesi", "pazartesileri"), ("Salı", "salıları"), ("Çarşamba", "çarşambaları"), ("Perşembe", "perşembeleri"),
        ("Cuma", "cumaları"), ("Cumartesi", "cumartesileri"), ("Pazar", "pazarları")]
ACTS = [("yüzmeye gidiyorsun", "yüzme"), ("gitar dersin var", "gitar dersi"), ("pazara gidiyorsun", "pazar alışverişi"),
        ("annenleri ziyaret ediyorsun", "annenleri ziyaret"), ("koşuya çıkıyorsun", "koşu"),
        ("İngilizce kursun var", "İngilizce kursu"), ("resim atölyesine gidiyorsun", "resim atölyesi")]


def _weekly(rng):
    day, pl = rng.choice(DAYS)
    act, noun = rng.choice(ACTS)
    return Fact("weekly", f"{pl} {act}", {pl},
                use=[(f"{day} akşamı arkadaşlar buluşalım diyor, uygun muyum?",
                      f"{_cap(pl)} {act}; buluşmayı {noun} sonrasına koyabilir ya da başka bir gün seçebilirsin."),
                     (f"Bu haftanın {day.lower()} gününü planlayalım",
                      f"{_cap(pl)} {act}, onu baştan yerleştirelim; geri kalan saatleri ona göre doldururuz.")],
                recall=[(f"{day} günleri neyim vardı?", f"{_cap(pl)} {act}.")])


RELS = [("annenin", "Annemin", "Annem"), ("babanın", "Babamın", "Babam"), ("kardeşinin", "Kardeşimin", "Kardeşim"),
        ("en yakın arkadaşının", "En yakın arkadaşımın", "En yakın arkadaşım")]


def _birthday(rng):
    poss, myg, my = rng.choice(RELS)
    d = rng.randint(1, 28)
    mon, loc = rng.choice(MONTHS)
    date = f"{d} {mon}"
    gift = rng.choice(["sevdiği bir çiçek ve el yazısıyla bir not her zaman güzeldir",
                       "birlikte bir kahvaltı ya da yemek en güzel hediye olabilir",
                       "küçük ama kişisel bir hediye, mesela bir fotoğraf albümü, çok anlamlı olur"])
    return Fact("birthday", f"{poss} doğum günü {date}", {date},
                use=[(f"{my} için bir hediye fikri lazım",
                      f"Doğum günü {date}{loc}; onu da düşünerek: {gift}.")],
                recall=[(f"{myg} yaş günü hangi tarihteydi?",
                         f"{date}." + rng.choice([" İstersen bir gün önceden hatırlatıcı kuralım mı?", ""]))])


DIETS = [("fıstığa alerjin var", "Fıstığa alerjin olduğu için onu dışarıda tutalım", "meyve tabağı ve yoğurtlu bir sos iyi olur"),
         ("laktoz hassasiyetin var", "Laktoz hassasiyetin olduğu için süt ürünlerini dışarıda tutalım",
          "humus ve sebze çubukları iyi olur"),
         ("acı yemeyi sevmiyorsun", "Acıyı sevmediğin için baharatı hafif tutalım", "fırında sebze ve sade yoğurt iyi olur"),
         ("glütensiz besleniyorsun", "Glütensiz beslendiğin için ekmek ve makarnayı dışarıda tutalım",
          "pirinç patlağı ve humus iyi olur")]


def _diet(rng):
    note, why, sugg = rng.choice(DIETS)
    return Fact("diet", note, set(),
                use=[(q, f"{why}: {sugg}.") for q in ["Akşam için bir atıştırmalık öner",
                                                     "Film gecesi için ne hazırlasam?",
                                                     "Hafif bir ara öğün önerir misin?"]])


WAKE = [("altıda", "ona", "on"), ("yedide", "on bire", "on bir"), ("sekizde", "gece yarısına", "on iki")]


def _wake(rng):
    w, until, _ = rng.choice(WAKE)
    return Fact("wake", f"sabahları {w} kalkıyorsun", {w},
                use=[("Bu akşam kaçta yatsam iyi olur?",
                      f"Sabahları {w} kalktığına göre en geç saat {until} kadar yatarsan sekiz saat uyumuş olursun.")],
                recall=[("Sabahları kaçta kalkıyordum?", f"{_cap(w)}.")])


GOALS = [("İspanyolca öğreniyorsun", "İspanyolca öğrendiğine göre", "bugün on yeni kelime öğrenmeye ne dersin?"),
         ("maraton için çalışıyorsun", "Maratona hazırlandığına göre", "bugün kısa ve rahat bir koşu, akşam da esneme olsun."),
         ("keman çalışıyorsun", "Keman çalıştığına göre", "bugün yirmi dakika ara vermeden çalmaya ne dersin?"),
         ("yazılım öğreniyorsun", "Yazılım öğrendiğine göre", "bugün küçük bir alıştırma programı yazmaya ne dersin?")]


def _goal(rng):
    note, lead, sugg = rng.choice(GOALS)
    return Fact("goal", note, set(),
                use=[(q, f"{lead} {sugg}") for q in ["Bugün kendime küçük bir hedef koymak istiyorum",
                                                     "Bugün ne yapsam kendimi geliştirmiş olurum?"]],
                recall=[("Şu sıralar ne öğreniyordum, hatırlıyor musun?", f"{_cap(note)}.")])


def _city(rng):
    c, loc = rng.choice(CITIES)
    return Fact("city", f"{loc} yaşıyorsun", {c},
                use=[("Hafta sonu için bir öneri ver",
                      f"{loc} yaşadığına göre şehrinde hiç gitmediğin bir semti keşfetmeye ne dersin?")],
                recall=[("Hangi şehirde yaşadığımı biliyor musun?", f"{loc} yaşıyorsun.")])


def _age(rng):
    n = rng.randint(19, 70)
    return Fact("age", f"{n} yaşındasın", {f"{n} yaş"},
                recall=[("Kaç yaşında olduğumu biliyor musun?", f"{n} yaşındasın.")])


FACTS = {"child": _child, "partner": _partner, "pet": _pet, "drink": _drink, "weekly": _weekly,
         "birthday": _birthday, "diet": _diet, "wake": _wake, "goal": _goal, "city": _city, "age": _age}
ALL_VALUES = (set(KID_NAMES) | set(PARTNER_NAMES) | set(PET_NAMES) | {c for c, _ in CITIES}
              | {f"{d} {m}" for d in range(1, 29) for m, _ in MONTHS})
USE_KINDS = [k for k in FACTS if k != "age"]                    # notu kullanan sorusu olanlar
RECALL_KINDS = [k for k in FACTS if k not in ("drink", "diet")]  # doğrudan hatırlatma sorusu olanlar
TYPES = {"kullan": 0.5, "hatirla": 0.15, "alakasiz": 0.2, "yok": 0.15}
THANKS_T2 = (["Süpersin, sağ ol", "Teşekkürler", "Harika fikir"],
             ["Ne demek! Seni tanıdıkça işim kolaylaşıyor.", "Ne demek, aklında olsun diye buradayım.",
              "Sevindim! Başka bir şey lazım olursa söyle."])


@dataclass
class MemoryDialog:
    cat: str
    line: str                       # "Hafıza: …" (satır sonu yok)
    turns: list[tuple[str, str]]
    used: Fact | None               # cevapta kullanılan not (kullan/hatirla)
    facts: list[Fact]               # satırdaki notlar


def memory_dialog(rng: random.Random, typ: str | None = None) -> MemoryDialog:
    from .personality import personality_turns
    typ = typ or rng.choices(list(TYPES), weights=list(TYPES.values()))[0]
    kinds = list(FACTS)
    if typ in ("kullan", "hatirla"):
        main = FACTS[rng.choice(USE_KINDS if typ == "kullan" else RECALL_KINDS)](rng)
        others = [FACTS[k](rng) for k in rng.sample([k for k in kinds if k != main.kind], rng.choice([0, 1, 1, 2]))]
        facts = others + [main]
        rng.shuffle(facts)
        line = memory_line([f.note for f in facts])
        facts = [f for f in facts if f.note in line[len(PREFIX):].split(SEP)]
        if main not in facts:  # sığmadıysa yalnızca ana not
            facts, line = [main], memory_line([main.note])
        q, a = rng.choice(main.use if typ == "kullan" else main.recall)
        if typ == "hatirla":
            a = " ".join(p for p in (a, rng.choice(RECALL_TAIL)) if p)
        turns = [(q, a.strip())]
        if typ == "kullan" and rng.random() < 0.2:
            turns.append((rng.choice(THANKS_T2[0]), rng.choice(THANKS_T2[1])))
        return MemoryDialog(typ, line, turns, main, facts)
    facts = [FACTS[k](rng) for k in rng.sample(kinds, rng.choice([1, 2, 2, 3]))]
    line = memory_line([f.note for f in facts])
    facts = [f for f in facts if f.note in line[len(PREFIX):].split(SEP)]
    if typ == "yok":
        have = {f.kind for f in facts}
        q, a = ABSENT[rng.choice([k for k in ABSENT if k not in have])]
        return MemoryDialog(typ, line, [(q, f"{a} {rng.choice(ABSENT_ASK)}")], None, facts)
    # alakasiz: hafıza satırı var ama soru notlarla ilgisiz; cevap notları anmaz
    _, turns = personality_turns(rng)
    return MemoryDialog(typ, line, turns, None, facts)
