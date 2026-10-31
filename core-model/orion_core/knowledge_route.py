"""Bilgi yönlendirme diyalogları (Tur 22): bilgi sorularında beyin uydurmaz; kaynağa yönelir.

42M'lik sohbet beyni binlerce olguyu ezberleyemez (MKQA denemesi: kendinden emin yanlış cevaplar). Doğru davranış:
"kesin bilmiyorum; uydurmak istemem" + kaynak önerisi (bilgi defteri, Vikipedi, araştırma, yardımcı yapay zeka) ya da
sahibine sormak. Konular kurgusal ve geniş; sınav soruları ve yakın biçimleri dialogs._turns_ok ile atılır.
"""
from __future__ import annotations

import random

SUBJECTS = [
    "Mars'taki en yüksek dağ", "ilk buharlı tren", "Japonca'daki yazı sistemleri", "kutup ayılarının ömrü",
    "Kapadokya'daki peri bacaları", "Mozart'ın son eseri", "dünyanın en derin gölü", "ilk cep telefonu",
    "Hitit başkenti", "arıların iletişimi", "Van kedisinin gözleri", "kuantum dolanıklığı", "Nil'in kaynağı",
    "Ege'deki en büyük ada", "ilk Türk romanı", "zeytin ağacının ömrü", "Saturn'ün halkaları", "Göbeklitepe'nin yaşı",
    "çayın Türkiye'ye gelişi", "ahtapotların kalp sayısı", "Everest'in ilk tırmanışı", "kahvenin keşfi",
    "Orhun yazıtları", "yıldırımın sıcaklığı", "balinaların şarkıları", "ilk bilgisayar programcısı",
]
ASK = ["{s} hakkında ne biliyorsun", "Bana {s} hakkında bilgi verir misin", "{s} konusunu anlatsana",
       "Merak ettim, {s} ile ilgili bildiğin var mı", "{s} neydi, hatırlıyor musun", "Kısaca {s} nedir"]
ANSWER = [
    "Bunu kesin bilmiyorum ve uydurmak istemem. Vikipedi'ye bakayım mı, yoksa sana araştırıp mı getireyim?",
    "Açık konuşayım: aklımda net bir bilgi yok. İstersen önce bilgi defterime bakayım, orada yoksa araştırayım.",
    "Emin olmadığım bir şeyi doğruymuş gibi söylemem. Hemen kaynağından bakayım mı?",
    "Bu konuda kafamda yalnızca belirsiz şeyler var; yanlış bilgi vermeyeyim. Araştırayım mı, sonra beraber okuruz?",
    "Bilmiyorum ama merak ettim! Vikipediye bakıp sana kısa bir özet getireyim mi?",
]
FOLLOW_YES = ["Evet, bak", "Olur, araştır", "Bak bakalım", "Tamam, getir"]
FOLLOW_YES_A = [
    "Tamam, bakıyorum. Bulduğumu kaynağıyla birlikte söyleyeceğim; emin olmadığım yer olursa bunu açıkça yazacağım.",
    "Peki, araştırıyorum. Kısa bir özet ve kaynağını getireceğim.",
]
FOLLOW_NO = ["Gerek yok", "Boş ver, sonra bakarız", "Yok, önemli değil"]
FOLLOW_NO_A = ["Peki. Aklına gelirse yeniden sorman yeter.", "Tamam, kenara not ettim; istersen sonra bakarız."]


def route_turns(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    """(kategori, turlar): tek tur ya da evet/hayır devamlı iki tur."""
    q = rng.choice(ASK).format(s=rng.choice(SUBJECTS))
    q = q[0].upper() + q[1:] + "?"
    turns = [(q, rng.choice(ANSWER))]
    r = rng.random()
    if r < 0.3:
        turns.append((rng.choice(FOLLOW_YES), rng.choice(FOLLOW_YES_A)))
        return "evet", turns
    if r < 0.45:
        turns.append((rng.choice(FOLLOW_NO), rng.choice(FOLLOW_NO_A)))
        return "hayir", turns
    return "tek", turns
