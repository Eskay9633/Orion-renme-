"""Tokenizer eğitimi için Türkçe sözcük + ek üretici (ünlü uyumuyla).

Şablon verisi dar olduğu için tek başına ~2.400 parçada doyar. Bu sözlük, kullanıcının
şablon dışında kullanacağı sözcüklerin de makul alt parçalara bölünmesini sağlar.
Yalnızca tokenizer eğitiminde kullanılır; model hedeflerine girmez.
"""
from __future__ import annotations

import random

NOUNS = """
ev iş okul araba telefon bilgisayar kitap defter kalem masa sandalye kapı pencere oda mutfak banyo bahçe
şehir köy yol sokak cadde mahalle ülke dünya deniz göl nehir dağ orman ağaç çiçek yaprak taş kum toprak
su ekmek süt peynir yumurta et tavuk balık sebze meyve elma armut muz portakal limon domates biber soğan
patates pirinç makarna çorba yemek kahvaltı akşam sabah öğle gece gün hafta ay yıl saat dakika saniye
zaman tarih takvim program plan proje görev toplantı rapor sunum dosya belge not liste kart grafik
para fiyat fatura kira maaş bütçe harcama hesap banka kredi borç ödeme alışveriş market mağaza ürün
sipariş kargo teslimat adres numara mesaj e-posta arama haber bilgi soru cevap fikir konu sorun çözüm
anne baba kardeş abla ağabey çocuk bebek arkadaş komşu müdür öğretmen öğrenci doktor hemşire avukat
mühendis tasarımcı yazılımcı yönetici müşteri ekip takım şirket ofis toplum aile insan kişi kadın erkek
sağlık hastane ilaç vitamin randevu muayene tahlil ameliyat ağrı ateş grip öksürük uyku spor yürüyüş
koşu yüzme bisiklet futbol basketbol maç oyun film dizi müzik şarkı konser tiyatro sinema müze sergi
tatil otel uçak tren otobüs metro taksi bilet bavul pasaport vize plaj kamp yürüyüş gezi rota harita
hava yağmur kar rüzgar güneş bulut sıcaklık soğuk sıcak mevsim ilkbahar yaz sonbahar kış
yazılım uygulama sistem sunucu veri tabanı model yapay zeka ajan araç komut kod hata test sürüm güncelleme
pil şarj ekran kamera depolama bellek ağ internet bağlantı ayar şifre güvenlik yedek bulut hesap profil
ders sınav ödev kurs eğitim dil İngilizce Türkçe matematik fizik kimya tarih coğrafya edebiyat
hediye doğum günü düğün bayram tören parti davet misafir ziyaret yolculuk alarm hatırlatıcı zamanlayıcı
sağlayıcı kota istek token lisans kaynak araştırma özet analiz ölçüm metrik hedef sonuç başarı
""".split()

VERBS = """
gel git yap al ver bak gör söyle yaz oku dinle izle ara bul sor anla düşün bil öğren öğret çalış dinlen
uyu kalk otur koş yürü yüz başla bitir dur bekle hatırla unut kaydet sil ekle çıkar değiştir düzelt
kontrol_et özetle araştır denetle göster gönder paylaş indir yükle kur aç kapa ödemek temizle hazırla
pişir ye iç sat öde topla planla tasarla ölç hesapla karşılaştır seç onayla reddet sakla yenile
""".split()

FRONT = set("eiöü")
BACK = set("aıou")
HARD = set("çfhkpsşt")


def _last_vowel(w: str) -> str:
    for ch in reversed(w):
        if ch in FRONT or ch in BACK:
            return ch
    return "e"


def _two(w: str, front: str, back: str) -> str:
    return front if _last_vowel(w) in FRONT else back


def _four(w: str) -> str:
    v = _last_vowel(w)
    return {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}[v]


def noun_forms(w: str) -> list[str]:
    hard = w[-1] in HARD
    vow = w[-1] in FRONT | BACK
    pl = w + _two(w, "ler", "lar")
    loc = w + ("t" if hard else "d") + _two(w, "e", "a")
    abl = w + ("t" if hard else "d") + _two(w, "en", "an")
    dat = w + ("y" if vow else "") + _two(w, "e", "a")
    acc = w + ("y" if vow else "") + _four(w)
    gen = w + ("n" if vow else "") + _four(w) + "n"
    poss1 = w + ("m" if vow else _four(w) + "m")
    with_ = w + ("yl" if vow else "l") + _two(w, "e", "a")
    return [w, pl, loc, abl, dat, acc, gen, poss1, with_, pl + _four(pl) + "m", pl + _four(pl)]


def verb_forms(v: str) -> list[str]:
    v = v.replace("_", " ")
    if " " in v:
        return [v, v + "ti", v + "meli", v + "ecek"]
    e2 = _two(v, "e", "a")
    i4 = _four(v)
    vow = v[-1] in FRONT | BACK
    return [
        v, v + "m" + e2 + "k", v + ("y" if vow else "") + i4 + "n", v + ("y" if vow else "") + i4 + "niz",
        v + ("t" if v[-1] in HARD else "d") + i4, v + "m" + _two(v, "iş", "ış"),
        v + ("y" if vow else "") + _two(v, "ecek", "acak"), v + "m" + e2 + "l" + i4,
        v + "s" + i4 + "n", v + ("r" if vow else i4 + "r"), v + ("y" if vow else "") + i4 + "yor",
    ]


def lexicon_sentences(n: int, seed: int = 7) -> list[str]:
    """Rastgele sözcük dizileri (dilbilgisel değil; yalnızca alt parça istatistiği için)."""
    rng = random.Random(seed)
    nouns = [f for w in NOUNS for f in noun_forms(w)]
    verbs = [f for v in VERBS for f in verb_forms(v)]
    out = []
    for _ in range(n):
        k = rng.randint(2, 6)
        words = [rng.choice(nouns) for _ in range(k - 1)] + [rng.choice(verbs)]
        out.append(" ".join(words))
    return out
