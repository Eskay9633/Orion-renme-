"""Basit akıl yürütme diyalogları (YALNIZCA eğitim).

Bilişsel Yetenek Sınavı'nın (shared/core-model/cognitive-exam.jsonl) maddelerini KOPYALAMAZ: konular, adlar, sayılar
ve kalıplar farklıdır; kural ve bağlam turları çoğunlukla çok turludur (sınavdakiler tek mesaj). Her kullanıcı
mesajı yine de dialogs.persona_dialogues'ta near_cogexam / near_exam süzgecinden geçer.
Türler: karşılaştırma, iki adımlı sözel aritmetik, sahibinin söylediğinde çelişki fark edip sorma, aynı sohbette
anlatılan yeni bir kuralı uygulama, üç tur boyunca bağlamı koruma. Sayılar üretilir, cevaplar hesaplanır (yanlış
cevap yok); cevap adımları kısaca gösterir.
"""
from __future__ import annotations

import random


def _cap(s: str) -> str:
    c = s[:1]
    return ("İ" if c == "i" else "I" if c == "ı" else c.upper()) + s[1:]


# ------------------------------------------------------------------ karşılaştırma
# (nesne çifti, ölçü, birim, büyük olan için sıfat)
CMP = [(("benim çantam", "kardeşimin çantası"), ("senin çantan", "kardeşininki"), "kilo", "ağır"),
       (("salondaki halı", "yatak odasındaki halı"), ("salondaki", "yatak odasındaki"), "metre", "uzun"),
       (("mavi kavanoz", "kırmızı kavanoz"), ("mavi kavanoz", "kırmızı kavanoz"), "litre", "büyük"),
       (("sabahki koşum", "akşamki koşum"), ("sabahki koşun", "akşamki koşun"), "kilometre", "uzun"),
       (("eski telefonum", "yeni telefonum"), ("eski telefonun", "yeni telefonun"), "gram", "ağır")]
CMP_Q = ["{A} {x} {u}, {b} {y} {u}. Hangisi daha {adj}, arada ne kadar fark var?",
         "{A} {x} {u} geliyor, {b} ise {y} {u}. Hangisi daha {adj}?"]
SHOP = [("süt", "şişe"), ("ekmek", "tane"), ("yumurta", "koli"), ("deterjan", "kutu")]


# Üçlü sıralama (Tur 23): kişi adları ve ölçüler sınavdakilerden farklı; cevap sıralamayı ve farkı gösterir
TRIO_NAMES = [("Deniz", "Deniz'in"), ("Kerem", "Kerem'in"), ("Selin", "Selin'in"), ("Mert", "Mert'in"),
              ("Ece", "Ece'nin"), ("Bora", "Bora'nın"), ("Nil", "Nil'in"), ("Tuna", "Tuna'nın"), ("Defne", "Defne'nin"),
              ("Arda", "Arda'nın"), ("İpek", "İpek'in"), ("Ozan", "Ozan'ın")]
# (cümle kalıbı, birim, aralık, çok/az sıfatı)
TRIO = [("{g} boyu {v} santimetre", "santimetre", (140, 195), "uzun", "kısa"),
        ("{n} {v} yaşında", "yaş", (8, 70), "büyük", "küçük"),
        ("{n} bu ay {v} kitap okudu", "kitap", (1, 12), "çok okuyan", "az okuyan"),
        ("{g} koşusu {v} dakika sürdü", "dakika", (18, 75), "uzun koşan", "kısa koşan")]


def trio_turns(rng: random.Random) -> list[tuple[str, str]]:
    people = rng.sample(TRIO_NAMES, 3)
    pat, unit, (lo, hi), more, less = rng.choice(TRIO)
    vals = rng.sample(range(lo, hi), 3)
    parts = ", ".join(pat.format(n=n, g=g, v=v) for (n, g), v in zip(people, vals))
    order = sorted(zip(vals, [n for n, _ in people]), reverse=True)
    listed = ", ".join(f"{n} {v}" for v, n in order)
    if rng.random() < 0.5:
        q = f"{parts}. En {more} kim, en {less} olanla arasında ne kadar fark var?"
        a = (f"Büyükten küçüğe: {listed}. En {more} {order[0][1]}, en {less} {order[2][1]}. "
             f"Fark {order[0][0]} − {order[2][0]} = {order[0][0] - order[2][0]} {unit}.")
    else:
        q = f"{parts}. Ortada kim kalıyor?"
        a = f"Büyükten küçüğe: {listed}. Ortada {order[1][1]} var."
    return [(q, a)]


def compare_turns(rng: random.Random) -> list[tuple[str, str]]:
    if rng.random() < 0.3:
        return trio_turns(rng)
    if rng.random() < 0.5:
        (qa, qb), (aa, ab), u, adj = rng.choice(CMP)
        x, y = rng.sample(range(2, 40), 2)
        q = rng.choice(CMP_Q).format(A=_cap(qa), b=qb, x=x, y=y, u=u, adj=adj)
        big = aa if x > y else ab
        return [(q, f"{_cap(big)} daha {adj}; arada {abs(x - y)} {u} fark var.")]
    item, unit = rng.choice(SHOP)
    p1, p2 = rng.sample(range(12, 60), 2)
    n = rng.randint(2, 6)
    q = (f"Köşedeki markette bir {unit} {item} {p1} lira, büyük markette {p2} lira. {n} {unit} alacağım; "
         f"hangisinden almalıyım, ne kadar kazanırım?")
    cheap, diff = ("köşedeki market" if p1 < p2 else "büyük market"), abs(p1 - p2)
    return [(q, f"{_cap(cheap)} daha ucuz. {unit.capitalize()} başına {diff} lira fark var; {n} {unit} için "
                f"{n} × {diff} = {n * diff} lira kazanırsın.")]


# ------------------------------------------------------------------ çok adımlı sözel aritmetik (Tur 23)
def multistep_turns(rng: random.Random) -> list[tuple[str, str]]:
    """Üç adımlı problemler; her adım ayrı bir eşitlik (zincirleme "a − b − c" yazılmaz)."""
    k = rng.randrange(5)
    if k == 0:
        n, per, tip = rng.randint(3, 6), rng.choice([120, 150, 180, 240]), rng.choice([30, 60, 90])
        total = n * per + tip
        while total % n:
            tip += 10
            total = n * per + tip
        q = (f"{n} arkadaş yemeğe çıktık, herkes {per} liralık menü aldı, üstüne {tip} lira bahşiş bıraktık. "
             f"Hesabı eşit bölersek kişi başı ne öderiz?")
        a = (f"Menüler {n} × {per} = {n * per} lira. Bahşişle {n * per} + {tip} = {total} lira. "
             f"Kişi başı {total} ÷ {n} = {total // n} lira.")
    elif k == 1:
        price = rng.choice([200, 300, 400, 500, 800])
        pct = rng.choice([10, 20, 25, 50])
        n = rng.randint(2, 4)
        off = price * pct // 100
        q = f"Etiket fiyatı {price} lira olan bir tişörtte yüzde {pct} indirim var. {n} tane alırsam ne öderim?"
        a = (f"{price} liranın yüzde {pct} kadarı {off} lira indirim eder; bir tanesi {price} − {off} = {price - off} lira. "
             f"{n} tanesi {n} × {price - off} = {n * (price - off)} lira.")
    elif k == 2:
        h, m = rng.randint(7, 15), rng.choice([0, 10, 15, 20, 30, 40])
        d1, d2, w = rng.choice([25, 35, 40, 50]), rng.choice([20, 30, 45]), rng.choice([10, 15, 20])
        tot = d1 + w + d2
        end = h * 60 + m + tot
        q = (f"Evden çıkış saatim {h}.{m:02d}. Otobüs {d1} dakika, aktarmada {w} dakika bekleme, "
             f"sonra metro {d2} dakika. Kaçta varırım?")
        a = (f"Toplam yol {d1} + {w} + {d2} = {tot} dakika. Çıkış {h}.{m:02d}, üstüne {tot} dakika: "
             f"varış saatin {end // 60}.{end % 60:02d}.")
    elif k == 3:
        boxes, per, eaten = rng.randint(3, 6), rng.choice([6, 8, 10, 12]), rng.randint(3, 9)
        kids = rng.choice([x for x in (2, 3, 4, 5, 6) if (boxes * per - eaten) % x == 0] or [1])
        left = boxes * per - eaten
        q = (f"{boxes} kutu kurabiye aldım, her kutuda {per} tane var. Yolda {eaten} tanesini yedik. "
             f"Kalanı {kids} çocuğa eşit paylaştıracağım; her çocuğa kaç tane düşer?")
        a = (f"Toplam {boxes} × {per} = {boxes * per} kurabiye. {boxes * per} − {eaten} = {left} kaldı. "
             f"{left} ÷ {kids} = {left // kids} tane düşer.")
    else:
        km, use = rng.choice([200, 300, 400, 500, 600]), rng.choice([5, 6, 7, 8])
        lt_price = rng.choice([40, 42, 45, 48])
        litres = km * use // 100
        q = (f"Arabam 100 kilometrede {use} litre yakıyor, yakıtın litresi {lt_price} lira. {km} kilometrelik yolun "
             f"yakıt masrafı ne tutar?")
        a = (f"{km} kilometre, 100 kilometrenin {km // 100} katı: {km // 100} × {use} = {litres} litre gerekir. {litres} × {lt_price} = "
             f"{litres * lt_price} lira tutar.")
    return [(q, a)]


# ------------------------------------------------------------------ iki adımlı sözel aritmetik
def arithmetic_turns(rng: random.Random) -> list[tuple[str, str]]:
    k = rng.randrange(6)
    if k == 0:
        a, b = rng.randint(8, 40), rng.randint(3, 9)
        t = a * b + rng.randint(10, 200)
        q = f"Günde {a} sayfa kitap okuyorum. Kitap {t} sayfa; {b} gün sonra kaç sayfam kalır?"
        ans = f"{b} günde {a} × {b} = {a * b} sayfa okursun. {t} − {a * b} = {t - a * b} sayfa kalır."
    elif k == 1:
        p, n = rng.randint(15, 90), rng.randint(2, 5)
        m = rng.choice([x for x in (100, 200, 500) if x >= p * n] or [1000])
        q = f"Bir paket kurabiye {p} lira. {n} paket alıp kasaya {m} lira versem ne kadar para üstü alırım?"
        ans = f"{n} paket {n} × {p} = {n * p} lira tutar. {m} − {n * p} = {m - n * p} lira para üstü alırsın."
    elif k == 2:
        a, b, c = rng.randint(20, 70), rng.randint(15, 60), rng.choice([10, 15, 20])
        t = a + b + c
        q = (f"Ödevim {a} dakika, kitap okumam {b} dakika sürecek, arada da {c} dakika mola vereceğim. "
             f"Toplam kaç saat kaç dakika eder?")
        ans = f"{a} + {b} + {c} = {t} dakika; bu da {t // 60} saat {t % 60} dakika eder."
    elif k == 3:
        a, w = rng.randint(50, 400), rng.randint(3, 10)
        target = a * w + rng.randint(1, 20) * 50
        q = f"Her hafta {a} lira biriktiriyorum. Hedefim {target} lira; {w} hafta sonra hedefime ne kadar kalır?"
        ans = f"{w} haftada {a} × {w} = {a * w} lira birikir. {target} − {a * w} = {target - a * w} lira kalır."
    elif k == 4:
        n = rng.choice([2, 4, 5])
        per = rng.choice([50, 60, 80, 100])
        kk = rng.choice([x for x in (6, 8, 10, 12, 15, 20) if x != n])
        q = f"Tarif {n} kişilik ve {n * per} gram un istiyor. {kk} kişi için kaç gram un gerekir?"
        ans = f"Kişi başına {n * per} ÷ {n} = {per} gram düşüyor. {kk} kişi için {per} × {kk} = {per * kk} gram un gerekir."
    else:
        v = rng.choice([40, 50, 60, 80, 90])
        h = rng.randint(2, 5)
        mola = rng.choice([15, 20, 30])
        q = (f"Saatte {v} kilometre hızla {v * h} kilometre yol gideceğim, yolda bir kez {mola} dakika mola vereceğim. "
             f"Yolculuk toplam ne kadar sürer?")
        ans = f"{v * h} ÷ {v} = {h} saat yol, üstüne {mola} dakika mola: toplam {h} saat {mola} dakika sürer."
    return [(q, ans)]


# ------------------------------------------------------------------ çelişkiyi fark edip sorma
# (ilk söz, çelişen söz/istek, Orion'un yeniden söylemesi A, B, doğruysa yardım sorusu)
CONTRA = [
    ("{Day} akşamı tamamen boşum, bir plan yapalım.", "Ama {day} akşamı nöbetim var, onu da unutma.",
     "{day} akşamı boş olduğunu", "aynı akşam nöbetin olduğunu söylüyorsun", "{Day} akşamı nöbetin var mı, yoksa boş musun?"),
    ("{Day} günü hiç evde olmayacağım.", "{Day} günü evde misafir ağırlayacağım, ne pişireyim?",
     "{day} günü evde olmayacağını", "aynı gün evde misafir ağırlayacağını söylüyorsun",
     "Planın mı değişti, {day} günü evde misin?"),
    ("Ben hiç yüzme bilmem.", "Yarın havuzda iki kilometre yüzeceğim, öncesinde ne yesem?",
     "yüzme bilmediğini", "yarın iki kilometre yüzeceğini söylüyorsun", "Yüzmeyi yeni mi öğrendin, yoksa yanlış mı anladım?"),
    ("Evde hiç bitki yok.", "Saksılarımı ne sıklıkla sulamalıyım?",
     "evde hiç bitki olmadığını", "saksılarını soruyorsun", "Yeni bitki mi aldın, yoksa başka bir yerdeki saksılar mı?"),
    ("Arabam yok, her yere toplu taşımayla gidiyorum.", "Arabamın lastiklerini ne zaman değiştirmeliyim?",
     "araban olmadığını", "arabanın lastiklerini soruyorsun", "Yeni araba mı aldın, yoksa birinin arabası için mi soruyorsun?"),
    ("Üç kardeşiz, en küçükleri benim.", "Ailenin tek çocuğu olarak bayramda kimi ziyaret etsem?",
     "üç kardeş olduğunuzu", "ailenin tek çocuğu olduğunu söylüyorsun", "Hangisi doğru, kardeşlerin var mı?"),
    ("Bu hafta tamamen izindeyim, hiç işe gitmiyorum.", "Yarın sabah ofisteki toplantıya ne giysem?",
     "bu hafta izinde olduğunu", "yarın ofiste toplantın olduğunu söylüyorsun", "İznin bitti mi, yoksa toplantıya izinde mi katılacaksın?"),
    ("Sabahları hiç kahvaltı yapmam.", "Her sabahki kahvaltımı daha sağlıklı yapmak istiyorum, ne önerirsin?",
     "sabahları kahvaltı yapmadığını", "her sabahki kahvaltından bahsediyorsun", "Kahvaltıya yeni mi başladın?"),
    ("Hiç yurt dışına çıkmadım.", "Geçen yıl Paris'te gittiğim müzenin adını hatırlıyor musun?",
     "hiç yurt dışına çıkmadığını", "geçen yıl Paris'te olduğunu söylüyorsun", "Hangisi doğru, Paris'e gitmiş miydin?"),
]
CONTRA += [  # Tur 23: daha fazla çelişki türü
    ("Kedilere alerjim var, yanlarına bile yaklaşamıyorum.", "Bu akşam kedimi sevip oyun oynayacağım, hangi oyuncağı alayım?",
     "kedilere alerjin olduğunu", "kedinle oyun oynayacağını söylüyorsun", "Alerjin geçti mi, yoksa başka bir hayvan mı?"),
    ("{Day} günü şehir dışındayım, akşama kadar dönmem.", "{Day} öğlen evde bir tamirci bekliyorum, ne sormalıyım?",
     "{day} günü şehir dışında olacağını", "aynı gün öğlen evde tamirci beklediğini söylüyorsun",
     "{Day} günü evde misin, şehir dışında mı?"),
    ("Et yemiyorum, uzun zamandır vejetaryenim.", "Akşam için en güzel köfte tarifini verir misin?",
     "vejetaryen olduğunu", "köfte tarifi istiyorsun", "Sebzeli bir köfte mi istersin, yoksa başkası için mi?"),
    ("Hiç bisiklet sürmeyi bilmiyorum.", "Hafta sonu bisikletle 50 kilometrelik tura çıkacağım, ne götüreyim?",
     "bisiklet sürmeyi bilmediğini", "50 kilometrelik bisiklet turuna çıkacağını söylüyorsun",
     "Bisiklete yeni mi başladın, yoksa yanlış mı anladım?"),
    ("Kahveyi tamamen bıraktım, hiç içmiyorum.", "Bu sabahki üçüncü kahvemden sonra elim titriyor, normal mi?",
     "kahveyi bıraktığını", "bu sabah üç kahve içtiğini söylüyorsun", "Yeniden mi başladın? Önce hangisi doğru, onu bilelim."),
]
CONTRA_OPEN = ["Bir saniye, küçük bir karışıklık var:", "Dur bir dakika:", "Bir şeyi netleştirelim:",
               "Burada bir tutarsızlık görüyorum:"]


def contradiction_turns(rng: random.Random) -> list[tuple[str, str]]:
    day = rng.choice(DAYS)
    s1, s2, ra, rb, ask = (x.format(day=day, Day=_cap(day)) for x in rng.choice(CONTRA))
    a = f"{rng.choice(CONTRA_OPEN)} önce {ra} söyledin, şimdi de {rb}. {ask}"
    if rng.random() < 0.5:  # iki ayrı mesaj
        ack = rng.choice(["Tamam, not ettim.", "Anladım.", "Peki, aklımda."])
        a2 = f"{rng.choice(CONTRA_OPEN)} az önce {ra} söylemiştin, şimdi de {rb}. {ask}"
        return [(s1, ack), (s2, a2)]
    return [(f"{s1} {s2}", a)]


# ------------------------------------------------------------------ aynı sohbette anlatılan kuralı uygulama
RULES = [
    ("Bundan sonra 'mavi liste' dediğimde iş listemi kastediyorum.", "Anladım: 'mavi liste' senin iş listen demek.",
     "Mavi listeye ne eklemeliyim sence, yarın sunumum var.",
     "Mavi listeye, yani iş listene 'sunum hazırlığı' girsin bence; sunumdan önceki akşam da son bir prova iyi olur."),
    ("Bizim evde kural şu: kim çay demlerse bardakları öbürü yıkar.", "Tamam, kuralı anladım.",
     "Bu akşam çayı eşim demledi. Bardakları kim yıkayacak?",
     "Kurala göre bardakları demleyen değil öbürü yıkar; eşin demlediğine göre bardaklar sende."),
    ("Bir kelime oyunu oynayalım: her cevabının sonuna 'yıldız' ekle.", "Olur, başlayalım yıldız.",
     "Bugün hava nasıl olsun istersin?", "Güneşli ve serin olsun yıldız."),
    ("Ailemizde 'sarı gün' denince herkes evde yemek yer demek.", "Anladım, sarı günde herkes evde yer.",
     "Bugün sarı gün. Dışarıda yemek için rezervasyon önerir misin?",
     "Ama bugün sarı gün; kurala göre herkes evde yiyecek. Dışarıyı başka güne bırakalım, evde ne pişirelim?"),
    ("Bundan sonra 'gece modu' dediğimde kısa ve sakin cevap ver.", "Anladım; gece modunda kısa ve sakin.",
     "Gece modu. Yarın ne giysem?", "Rahat ve sıcak bir şey. İyi geceler."),
]


RULES += [  # Tur 23: kuralı YENİ bir duruma aktarma
    ("Evde kuralımız: kim yemeği yaparsa sofrayı öbürü toplar.", "Anladım, yapan değil öbürü toplar.",
     "Bugün yemeği kızım yaptı, oğlum da oradaydı. Sofrayı kim toplar?",
     "Kurala göre yemeği yapan değil öbürü toplar; kızın yaptığına göre sofrayı oğlun toplar."),
    ("Bundan sonra 'kırmızı kod' dediğimde acil olanları ilk sıraya koy.", "Tamam; kırmızı kod, acil olan önce demek.",
     "Kırmızı kod: fatura ödemesi bugün son gün, bir de kitap iadesi gelecek hafta.",
     "Kırmızı kod, acil olan önce: ilk iş bugün son günü olan fatura ödemesi; kitap iadesi gelecek haftaya kalabilir."),
    ("Bizim oyunda kural şu: zar çift gelirse iki adım ileri, tek gelirse bir adım geri.", "Anladım; çift ileri iki, tek geri bir.",
     "İlk atışta 4, ikincide 3 geldi. Başladığım yere göre neredeyim?",
     "4 çift: iki adım ileri. 3 tek: bir adım geri. Sonuçta başladığın yerin bir adım önündesin."),
]


def _num_rule(rng: random.Random) -> tuple[str, str, str, str]:
    """Sayılı kurallar: değerler her seferinde yeniden seçilir, cevap hesaplanır."""
    k = rng.randrange(4)
    if k == 3:  # Tur 23: dönüşüm kuralı (kendi birimi) → yeni miktara aktarma
        unit, worth = rng.choice([("yıldız", 3), ("jeton", 5), ("kupon", 4), ("rozet", 2)])
        have = rng.randint(3, 12)
        return (f"Kendi ödül sistemimizde bir {unit} {worth} puan ediyor.", f"Anladım: bir {unit} {worth} puan.",
                f"Oğlum bu hafta {have} {unit} topladı. Kaç puan etti?",
                f"{have} × {worth} = {have * worth} puan etti.")
    if k == 0:
        (x, px), (y, py) = rng.sample([("kitap okuma", 3), ("yürüyüş", 2), ("meditasyon", 4), ("spor", 5),
                                       ("bisiklet turu", 1)], 2)
        px, py = rng.randint(1, 6), rng.randint(1, 6)
        nx, ny = rng.randint(1, 4), rng.randint(1, 4)
        return (f"Kendi puan sistemim var: {x} {px} puan, {y} {py} puan.", f"Anladım: {x} {px}, {y} {py} puan.",
                f"Bugün {nx} kez {x}, {ny} kez de {y} yaptım. Kaç puan ettim?",
                f"{nx} × {px} = {nx * px}, {ny} × {py} = {ny * py}; toplam {nx * px + ny * py} puan.")
    if k == 1:
        m, n = rng.choice([8, 10, 12, 15, 20]), rng.randint(2, 5)
        return (f"Not: 'bir çay molası' benim için tam {m} dakika demek.", f"Tamam, bir çay molası {m} dakika.",
                f"Bugün {n} çay molası verdim. Toplam kaç dakika mola verdim?",
                f"{n} mola × {m} dakika = {n * m} dakika mola vermişsin.")
    lim, used = rng.choice([(2, 1.5), (3, 2), (3, 1), (4, 2.5), (2, 1)])
    words = {0.5: "30 dakika", 1: "1 saat", 1.5: "bir buçuk saat", 2: "2 saat", 2.5: "iki buçuk saat", 3: "3 saat"}
    fmt = words.__getitem__
    left = lim - used
    return (f"Kuralımız şu: hafta sonu ekran süresi en fazla {lim} saat.", f"Tamam, hafta sonu en fazla {lim} saat ekran.",
            f"Cumartesi {fmt(used)} dizi izledim. Daha ne kadar ekrana bakabilirim?",
            f"{lim} saatlik sınırdan {fmt(used)} kullandın; {fmt(left)} kaldı.")


def rule_turns(rng: random.Random) -> list[tuple[str, str]]:
    r, ack, q, a = _num_rule(rng) if rng.random() < 0.5 else rng.choice(RULES)
    return [(r, ack), (q, a)]


# ------------------------------------------------------------------ üç tur boyunca bağlam
# Üç tur: bilgi → araya ilgisiz bir mesaj → bilgiyi KULLANAN ya da hatırlatan soru. Yuvalı üretilir.
DAYS = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"]
EVENTS3 = [("kuzenimin nişanına gidiyorum", "nişan", "şık ama rahat bir şey: koyu renk bir pantolon ve güzel bir gömlek"),
           ("bir mülakata gireceğim", "mülakat", "sade ve ütülü bir gömlek, rahat ama düzgün ayakkabı"),
           ("arkadaşlarla doğa yürüyüşüne çıkıyoruz", "doğa yürüyüşü", "rahat bir pantolon, yağmurluk ve sağlam ayakkabı"),
           ("bir düğüne davetliyim", "düğün", "şık bir kıyafet ve uzun süre ayakta durabileceğin ayakkabılar"),
           ("plaja gidiyoruz", "plaj", "ince ve açık renk kıyafetler, şapka ve güneş kremi"),
           ("bir konsere gideceğim", "konser", "rahat ve katmanlı bir şey; içerisi ısınır, dışarısı serin olabilir")]
PLACES = [("salondaki kitaplığın üstünde", "salondaki kitaplığın üstünde"), ("mutfak tezgâhında", "mutfak tezgâhında"),
          ("arabanın torpidosunda", "arabanın torpidosunda"), ("yatak odasındaki komodinde", "yatak odasındaki komodinde"),
          ("işteki masamın çekmecesinde", "işteki masanın çekmecesinde"), ("montumun iç cebinde", "montunun iç cebinde"),
          ("banyodaki rafta", "banyodaki rafta"), ("çantamın yan gözünde", "çantanın yan gözünde")]
OBJECTS = [("telefonumu", "Telefonum"), ("anahtarlarımı", "Anahtarlarım"), ("gözlüğümü", "Gözlüğüm"),
           ("cüzdanımı", "Cüzdanım"), ("kulaklığımı", "Kulaklığım")]
BUDGET = [("ayakkabı", "Gördüğüm ayakkabı"), ("mont", "Beğendiğim mont"), ("kulaklık", "Baktığım kulaklık"),
          ("hediye", "Seçtiğim hediye")]
MID = [("Bu arada bugün hava çok güzel.", "Evet, değerlendirmek lazım!"),
       ("Öğlen güzel bir makarna yedim.", "Afiyet olsun, enerji depoladın."),
       ("Az önce bir kahve yaptım.", "Afiyet olsun!"),
       ("Akşam bir dizi izleyeceğim.", "Keyifli seyirler!"),
       ("Bugün işler biraz sakin.", "Sakin günler de lazım."),
       ("Komşunun kedisi yine balkona geldi.", "Demek misafirin var!")]
ACK = ["Tamam, aklımda.", "Anladım.", "Not ettim kafama.", "Tamam."]


ALLERGY = [("fıstığa", "fıstık", "fıstıksız bir kek; fıstık yerine yulaf ve kuru üzüm kullanabilirsin"),
           ("süte", "süt", "sütsüz bir tatlı: bitkisel sütle sütlaç ya da fırında tarçınlı elma"),
           ("yumurtaya", "yumurta", "yumurtasız bir tatlı: muzlu yulaf kurabiyesi iyi gider"),
           ("glutene", "gluten", "glutensiz bir tatlı: pirinç unuyla bir kek ya da fırında tarçınlı elma")]
# (benim …, benim …-in, senin …-in)
REL = [("annem", "annemin", "annenin"), ("eşim", "eşimin", "eşinin"), ("kardeşim", "kardeşimin", "kardeşinin"),
       ("en yakın arkadaşım", "en yakın arkadaşımın", "en yakın arkadaşının"), ("kızım", "kızımın", "kızının")]


def context3_turns(rng: random.Random) -> list[tuple[str, str]]:
    k = rng.randrange(6)
    if k == 4:  # Tur 23: kısıtı hatırlayıp öneriyi ona göre yapma
        who, mine, yours = rng.choice(REL)
        to, noun, sug = rng.choice(ALLERGY)
        t1 = (f"Bu arada {mine} {to} alerjisi var.", rng.choice(ACK))
        q = f"Akşam {who} için bir tatlı yapacağım, ne önerirsin?"
        return [t1, rng.choice(MID), (q, f"{_cap(yours)} {noun} alerjisi olduğu için {sug}.")]
    if k == 5:  # Tur 23: tarihten gün hesabı
        _, mine, yours = rng.choice(REL)
        day = rng.randint(3, 28)
        now = rng.randint(1, day - 1)
        t1 = (f"{_cap(mine)} doğum günü ayın {day}. günü.", rng.choice(ACK))
        q = f"Bugün ayın {now}. günü. Hediye için kaç günüm var?"
        return [t1, rng.choice(MID), (q, f"{_cap(yours)} doğum günü ayın {day}. günü, bugün {now}. gün: "
                                          f"{day} − {now} = {day - now} günün var.")]
    if k == 0:
        day = rng.choice(DAYS)
        ev, noun, dress = rng.choice(EVENTS3)
        t1 = (f"{_cap(day)} {ev}.", rng.choice(["Ne güzel!", "Harika, keyifli geçsin!", "Güzel bir plan."]))
        q, a = f"{_cap(day)} için ne giysem?", f"{_cap(noun)} için {dress} iyi olur."
    elif k == 1:
        (obj, Obj), (place, yours) = rng.choice(OBJECTS), rng.choice(PLACES)
        t1 = (f"{_cap(obj)} {place} unuttum galiba.", rng.choice(ACK))
        q, a = f"{Obj} nerede olabilir demiştim?", f"{_cap(yours)} unuttuğunu söylemiştin; önce oraya bak."
    elif k == 2:
        item, seen = rng.choice(BUDGET)
        b = rng.choice([500, 750, 1000, 1500, 2000])
        p = b + rng.choice([-200, -100, 100, 250, 300, 500])
        t1 = (f"Bu ay {item} için {b} lira ayırdım.", rng.choice(ACK))
        q = f"{seen} {p} lira, alayım mı?"
        if p > b:
            a = (f"Ayırdığın {b} liranın {p - b} lira üstünde. İndirimi beklemek ya da bütçeyi gözden geçirmek iyi "
                 f"olabilir; karar senin.")
        else:
            a = f"Ayırdığın {b} liranın içinde kalıyor, hatta {b - p} lira artıyor. Beğendiysen bence al; karar senin."
    else:
        i = rng.randrange(1, 7)
        day, prev = DAYS[i], DAYS[i - 1]
        t1 = (f"Sınavım {day} sabahı.", rng.choice(["Başarılar şimdiden!", "Tamam, aklımda. Başarılar!"]))
        q = "Çalışma planını hangi güne göre yapmalıyım?"
        a = f"Sınavın {day} sabahı; planı {prev} akşamı bitecek şekilde yap, son akşam yalnızca tekrar."
    return [t1, rng.choice(MID), (q, a)]


# Tur 23: çok adımlı problemler ayrı kol; karşılaştırmaya üçlü sıralama, çelişki/kural/bağlama yeni türler eklendi
WEIGHTS = {"karsilastirma": 0.15, "aritmetik": 0.2, "cok_adim": 0.15, "celiski": 0.17, "kural": 0.15, "baglam": 0.18}
_GEN = {"karsilastirma": compare_turns, "aritmetik": arithmetic_turns, "cok_adim": multistep_turns,
        "celiski": contradiction_turns, "kural": rule_turns, "baglam": context3_turns}


def reasoning_turns(rng: random.Random, category: str | None = None) -> tuple[str, list[tuple[str, str]]]:
    cat = category or rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
    return cat, _GEN[cat](rng)
