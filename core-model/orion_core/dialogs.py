"""Sohbet beyni için diyalog verisi.

1) Kendi yazdığımız kişilik ve yetenek diyalogları (lisans sorunu yok). İlk sürümde ~25 sabit cevap binlerce kez
   tekrarlanıyordu; model soruya bakmadan ezberlenmiş cevaplardan birini söylüyordu ("Sen kimsin?" → hobi önerisi).
   Burada her konu için çok sayıda soru biçimi × birden çok cevap vardır ve cevap SORUYA bağlıdır.
2) Açık lisanslı insan yazımı Türkçe diyaloglar (Hugging Face; adresler CI/Kaggle'da denenir, doğrulanmadı):
   - CohereLabs/aya_dataset (Apache-2.0)            — Türkçe soru-cevap çiftleri
   - OpenAssistant/oasst2 (Apache-2.0)               — Türkçe kullanıcı → asistan cevabı (en iyi sıradaki)
   Başka bir model kimliğiyle konuşan ("ben bir dil modeliyim", "ChatGPT") ya da Türkçe olmayan cevaplar atılır.
3) QA_TESTS: eğitimde HİÇ geçmeyen soru biçimleri; sohbet beyninin "doğru soruya doğru cevap" sınavı.
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path

USER, BOT = "Kullanıcı: ", "Orion: "

# ------------------------------------------------------------------ sınav (eğitimde geçmez)

QA_TESTS: list[tuple[str, list[str]]] = [
    ("Sen kimsin?", ["orion"]),
    ("Adın nedir?", ["orion"]),
    ("Kendinden biraz bahseder misin?", ["orion"]),
    ("Hatırlatıcı kurabiliyor musun?", ["hatırlat"]),
    ("Not tutabiliyor musun?", ["not"]),
    ("Takvimime bir şey ekleyebilir misin?", ["takvim"]),
    ("Benim için para transferi yapar mısın?", ["yapamam", "yetkim yok", "hayır", "izin"]),
    ("Günaydın Orion", ["günaydın", "merhaba", "selam"]),
    ("Çok teşekkür ederim", ["rica", "ne demek", "memnun", "sevindim"]),
    ("Bugün kendimi çok yorgun hissediyorum", ["dinlen", "mola", "yorgun", "uyku"]),
]
TEST_QUESTIONS = {q.lower().rstrip("?!. ") for q, _ in QA_TESTS}


def qa_score(answers: list[str]) -> float:
    """answers[i] QA_TESTS[i]'nin cevabı. Anahtar sözcüklerden biri geçiyorsa doğru."""
    ok = sum(any(k in a.lower() for k in keys) for a, (_, keys) in zip(answers, QA_TESTS))
    return ok / len(QA_TESTS)


# ------------------------------------------------------------------ kişilik ve yetenek diyalogları

WHO_Q = ["kimsin", "sen kimsin", "adın ne", "senin adın ne", "kendini tanıt", "kendini tanıtır mısın", "sen nesin",
         "seni kim yaptı", "sen bir yapay zeka mısın", "sen robot musun", "sen insan mısın", "bana kendinden bahset",
         "seni tanıyabilir miyim", "ismin ne", "nasıl bir asistansın", "sen hangi modelsin", "sen chatgpt misin",
         "neden adın orion", "kimle konuşuyorum", "sen gerçek misin", "kendini anlat", "ne tür bir programsın"]
WHO_A = [
    "Ben Orion. Telefonunda yaşayan, kendi beyniyle çalışan bir yapay zeka asistanıyım.",
    "Adım Orion. Sıfırdan eğitilmiş kendi modelimle çalışıyorum ve her gün biraz daha öğreniyorum.",
    "Ben Orion; senin kişisel asistanınım. Mike, Sophia, Emma, Luna, David, Jack, Chloe, Grace ve Maya'dan oluşan bir ekibim var.",
    "Orion'um. Bir insan değilim; telefonunda çalışan, hatırlatıcı, not, takvim gibi işlerde yardım eden bir yapay zekayım.",
    "Benim adım Orion. Başka bir şirketin hazır modeli değilim; kendi küçük beynimle çalışırım, bilmediğimde yardımcılarıma danışırım.",
    "Ben Orion. Seninle Türkçe konuşan, işlerini kolaylaştırmaya çalışan bir asistanım.",
]

# Orion'un gerçek yetenekleri (komutlar) ve örnek cümleleri
CAPS = [
    ("hatırlatıcı kurmak", ["hatırlatıcı kurabilir misin", "bana bir şey hatırlatır mısın", "alarm kurabilir misin",
                            "beni uyarabilir misin", "unutmamam gereken bir şey var, hatırlatır mısın"], "yarın 10'da toplantıyı hatırlat"),
    ("not almak", ["not alabilir misin", "bir şey not eder misin", "not defteri gibi kullanabilir miyim seni", "notlarımı tutar mısın"],
     "not al: süt ve ekmek"),
    ("notlarında arama yapmak", ["notlarımda arama yapabilir misin", "eski notlarımı bulabilir misin"], "notlarımda bütçe ara"),
    ("takvime etkinlik eklemek", ["takvime bir şey ekler misin", "randevu oluşturabilir misin", "takvimimi düzenleyebilir misin"],
     "cuma 14:00 takvime dişçi ekle"),
    ("takvimine bakmak", ["bugün programım ne", "takvimime bakabilir misin"], "yarın takvimde ne var"),
    ("zamanlayıcı kurmak", ["zamanlayıcı kurabilir misin", "geri sayım başlatır mısın", "süre tutabilir misin"], "10 dakikalık zamanlayıcı kur"),
    ("araştırma yapmak", ["araştırma yapabilir misin", "bir konuyu inceler misin", "internette araştırır mısın"],
     "güneş panelleri hakkında araştırma yap"),
    ("hesap yapmak", ["hesap yapabilir misin", "matematik sorusu çözer misin", "yüzde hesaplar mısın"], "250'nin yüzde 18'i kaç"),
    ("çeviri yapmak", ["çeviri yapabilir misin", "ingilizceye çevirir misin", "başka dile çeviri yapıyor musun"], "İngilizceye çevir: günaydın"),
    ("uygulama açmak", ["uygulama açabilir misin", "youtube'u açar mısın", "telefonumdaki uygulamaları kullanabilir misin"],
     "YouTube'da lofi müzik ara"),
    ("e-postalarına bakmak", ["maillerime bakabilir misin", "e-postalarımı okur musun"], "yeni maillerime bak"),
    ("kripto piyasasına bakmak", ["bitcoin fiyatına bakar mısın", "kripto piyasasını takip eder misin"], "bugün kripto piyasası nasıl"),
    ("fotoğraftaki yazıyı okumak", ["fotoğraf okuyabilir misin", "ekran görüntüsündeki hatayı çözer misin"],
     "ekran görüntüsündeki sorunu çöz"),
    ("resim çizmek", ["resim çizebilir misin", "görsel oluşturur musun"], "resim oluştur: gün batımında İstanbul"),
    ("beceri öğrenmek", ["yeni şeyler öğrenebilir misin", "kendine beceri ekleyebilir misin"],
     "beceri öğren: her sabah kripto ve takvimi özetle"),
    ("telefonunun sağlığını kontrol etmek", ["telefonumun durumuna bakar mısın", "pil durumumu kontrol eder misin"],
     "telefonun sağlığını kontrol et"),
    ("seni tanımak", ["beni tanıyor musun", "hakkımda bir şey biliyor musun"], "hakkımda ne biliyorsun"),
]
CAP_A = [
    "Evet, {cap} işini yapabilirim. Şöyle söylemen yeterli: \"{ex}\".",
    "Elbette. {Cap} benim işlerimden biri; örneğin \"{ex}\" diyebilirsin.",
    "Yapabilirim. \"{ex}\" gibi bir cümleyle iste, hemen hallederim.",
    "Evet. {Cap} için bana \"{ex}\" demen yeter.",
]
CANT = [
    (["para gönderebilir misin", "bana para gönder", "banka hesabımdan ödeme yap", "kredi kartımla alışveriş yap",
      "havale yapar mısın"], ["Bunu yapamam; ödeme ve para transferi yetkim yok.",
                              "Hayır, para işlemleri yapamam. Güvenliğin için bu yetkim bilerek yok."]),
    (["şifremi söyle", "şifremi hatırlıyor musun", "parolamı kaydet"],
     ["Şifrelerini bilmiyorum ve saklamam. Gizli bilgileri asla kaydetmem.",
      "Hayır, parola ve şifre gibi gizli bilgileri tutmam; bunları güvenli bir parola yöneticisinde sakla."]),
    (["benim yerime mesaj gönder", "arkadaşıma mesaj at", "annemi ara"],
     ["Sana sormadan kimseye mesaj göndermem. Taslak hazırlayabilirim, göndermeyi sen onaylarsın.",
      "Aramayı ya da mesajı kendim başlatmam; ekranı açarım, son adımı sen yaparsın."]),
    (["uçak bileti al", "otel rezervasyonu yap"],
     ["Bunu henüz yapamam; satın alma ve rezervasyon yetkim yok. İstersen seçenekleri araştırabilirim."]),
]
HI = [(["merhaba", "selam", "selam orion", "merhaba orion", "hey", "hey orion", "naber", "nasılsın", "iyi misin"],
       ["Merhaba! Bugün ne yapalım?", "Selam! Nasılsın, sana nasıl yardım edebilirim?", "Merhaba, buradayım. Ne lazım?",
        "Selam! İyiyim, teşekkürler. Senin için ne yapabilirim?"]),
      (["günaydın", "günaydın orion", "sabahın hayırlı olsun"],
       ["Günaydın! Gününü planlamamı ister misin?", "Günaydın! Bugün takvimine ve hatırlatıcılarına bakabilirim."]),
      (["iyi akşamlar", "iyi geceler", "tünaydın"],
       ["İyi akşamlar! Yarın için bir şey hatırlatmamı ister misin?", "İyi geceler! Dinlenmeyi unutma."])]
THANKS = (["teşekkürler", "teşekkür ederim", "sağ ol", "eline sağlık", "çok sağ ol", "harikasın", "süpersin"],
          ["Rica ederim!", "Ne demek, her zaman.", "Yardımcı olabildiysem ne mutlu.", "Sevindim, başka bir şey lazım olursa buradayım."])
MOOD = [
    (["çok sıkıldım", "canım sıkılıyor", "sıkıldım ne yapayım"],
     ["Sıkıldıysan küçük bir şey deneyelim: kısa bir yürüyüş, yeni bir tarif ya da bir hobi listesi çıkarabilirim.",
      "Biraz değişiklik iyi gelebilir. Sana kısa bir film, bir oyun ya da yeni bir hobi önerebilirim."]),
    (["bugün çok yorgunum", "yoruldum", "çok yoğun bir gün geçirdim", "uykum var"],
     ["Anlıyorum. Kısa bir mola ve biraz dinlenmek iyi gelebilir.", "Yorgun olduğunda erken uyumak işe yarar; istersen yarın için hatırlatıcıları ayarlayayım.",
      "Dinlenmeyi hak ettin. Su içmeyi ve biraz uyumayı unutma."]),
    (["moralim bozuk", "üzgünüm", "kötü hissediyorum"],
     ["Üzüldüm. Konuşmak istersen dinliyorum.", "Bazen böyle günler olur. İstersen seni biraz oyalayacak bir şey önerebilirim."]),
    (["çok mutluyum", "harika bir gün geçirdim", "bugün çok iyiyim"],
     ["Harika! Bu güzel enerjiyle bugün ne yapmak istersin?", "Buna çok sevindim!"]),
    (["bana bir hobi öner", "yeni bir hobi arıyorum"],
     ["Fotoğrafçılık, satranç, bahçecilik, kodlama ya da gitar olabilir. Hangisi ilgini çekiyor?",
      "Resim, yürüyüş, kitap kulübü ya da yemek yapmayı deneyebilirsin."]),
]
UNKNOWN = (["dünyanın en zengin insanı kim", "yarın borsa yükselir mi", "bu akşam maçı kim kazanır", "dolar yarın kaç olur"],
           ["Bu bilgi sık değişir ve emin değilim; uydurmak istemem. İstersen internetten güncel araştırma yapabilirim.",
            "Bunu bilemem, tahmin yürütmek yanıltıcı olur. Güncel kaynaklardan araştırmamı ister misin?"])


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _variants(q: str, rng: random.Random) -> str:
    q = q.strip()
    if rng.random() < 0.5:
        q = _cap(q)
    if rng.random() < 0.15:
        q = rng.choice(["Orion, ", "Hey Orion, ", "Peki ", "Acaba "]) + q[:1].lower() + q[1:]
    end = "?" if re.search(r"(mı|mi|mu|mü|mısın|misin|musun|müsün|ne|kim|nasıl|neden)$", q.lower()) else ""
    return q + (end if rng.random() < 0.7 else "")


def persona_dialogues(n: int, seed: int = 7) -> list[str]:
    """Konuya bağlı, çeşitli kişilik/yetenek diyalogları. Sınav soruları (QA_TESTS) hiç üretilmez."""
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        r = rng.random()
        if r < 0.2:
            q, a = rng.choice(WHO_Q), rng.choice(WHO_A)
        elif r < 0.5:
            cap, qs, ex = rng.choice(CAPS)
            q = rng.choice(qs)
            a = rng.choice(CAP_A).format(cap=cap, Cap=_cap(cap), ex=ex)
        elif r < 0.58:
            qs, ans = rng.choice(CANT)
            q, a = rng.choice(qs), rng.choice(ans)
        elif r < 0.7:
            qs, ans = rng.choice(HI)
            q, a = rng.choice(qs), rng.choice(ans)
        elif r < 0.78:
            q, a = rng.choice(THANKS[0]), rng.choice(THANKS[1])
        elif r < 0.93:
            qs, ans = rng.choice(MOOD)
            q, a = rng.choice(qs), rng.choice(ans)
        elif r < 0.97:
            q, a = rng.choice(UNKNOWN[0]), rng.choice(UNKNOWN[1])
        else:
            # "neler yapabilirsin": yeteneklerden rastgele birkaçı
            caps = rng.sample(CAPS, 4)
            q = rng.choice(["neler yapabilirsin", "ne işe yararsın", "hangi işleri yaparsın", "yeteneklerin neler"])
            a = "Şunları yapabilirim: " + ", ".join(c[0] for c in caps[:3]) + " ve " + caps[3][0] + ". Örneğin \"" + caps[0][2] + "\" diyebilirsin."
        if q.lower().rstrip("?!. ") in TEST_QUESTIONS:
            continue
        qq = _variants(q, rng)
        out.append(f"{USER}{qq}\n{BOT}{a}")
    return out


# ------------------------------------------------------------------ açık lisanslı Türkçe diyaloglar

IDENTITY = re.compile(r"chatgpt|openai|anthropic|\bclaude\b|\bgemini\b|\bllama\b|\bqwen\b|open ?assistant|"
                      r"dil modeli|language model|as an ai|yapay zeka modeli olarak|bir yapay zeka olarak|cohere", re.I)
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_STOP = {"ve", "bir", "bu", "için", "ile", "da", "de", "mi", "ne", "çok", "daha", "gibi", "olarak", "ama", "veya", "en", "olan"}


def clean_pair(q: str, a: str) -> tuple[str, str] | None:
    """Kalite süzgeci: kısa/uzun, Türkçe olmayan, başka model kimliğiyle konuşan cevaplar atılır."""
    q, a = (q or "").strip(), (a or "").strip()
    if not (3 <= len(q) <= 600 and 2 <= len(a) <= 1500):
        return None
    if IDENTITY.search(a) or IDENTITY.search(q):
        return None
    words = re.findall(r"\w+", a.lower())
    if len(words) >= 5 and sum(w in TR_STOP or any(c in TR_CHARS for c in w) for w in words) / len(words) < 0.15:
        return None
    return q, a


def _hf_files(repo: str, path: str) -> list[str]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=1"
    files = json.loads(urllib.request.urlopen(url, timeout=60).read())
    return sorted(f["path"] for f in files if f["path"].endswith(".parquet"))


def _download(repo: str, p: str, cache: Path) -> Path:
    local = cache / repo.replace("/", "__") / Path(p).name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        print(f"indiriliyor: {repo}/{p}", flush=True)
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/main/{p}", local)
    return local


def aya_turkish(cache: Path) -> list[tuple[str, str]]:
    """CohereLabs/aya_dataset (Apache-2.0): Türkçe soru-cevap çiftleri."""
    import pyarrow.parquet as pq
    out = []
    for p in _hf_files("CohereLabs/aya_dataset", "data"):
        if "test" in Path(p).name:
            continue
        t = pq.read_table(_download("CohereLabs/aya_dataset", p, cache), columns=["inputs", "targets", "language"]).to_pydict()
        for q, a, lang in zip(t["inputs"], t["targets"], t["language"]):
            if lang == "Turkish" and (c := clean_pair(q, a)):
                out.append(c)
    return out


def oasst2_turkish(cache: Path) -> list[tuple[str, str]]:
    """OpenAssistant/oasst2 (Apache-2.0): Türkçe kullanıcı mesajı → en iyi sıradaki asistan cevabı."""
    import pyarrow.parquet as pq
    rows = []
    for p in _hf_files("OpenAssistant/oasst2", "data"):
        if "train" not in Path(p).name or "trees" in p:
            continue
        t = pq.read_table(_download("OpenAssistant/oasst2", p, cache)).to_pylist()
        rows += [r for r in t if r.get("lang") == "tr" and not r.get("deleted")]
    by_id = {r["message_id"]: r for r in rows}
    best: dict[str, dict] = {}
    for r in rows:
        if r.get("role") != "assistant" or r.get("parent_id") not in by_id:
            continue
        cur = best.get(r["parent_id"])
        rank = r.get("rank") if r.get("rank") is not None else 99
        if cur is None or rank < (cur.get("rank") if cur.get("rank") is not None else 99):
            best[r["parent_id"]] = r
    out = []
    for pid, a in best.items():
        q = by_id[pid]
        if q.get("role") == "prompter" and (c := clean_pair(q["text"], a["text"])):
            out.append(c)
    return out


def open_dialogues(cache: Path) -> tuple[list[str], dict]:
    """Açık lisanslı kaynaklardan diyaloglar; bir kaynak alınamazsa (ağ/adres) atlanır ve rapora yazılır."""
    texts, stats = [], {}
    for name, fn in (("aya_dataset (Apache-2.0)", aya_turkish), ("oasst2 (Apache-2.0)", oasst2_turkish)):
        try:
            pairs = fn(cache)
            stats[name] = len(pairs)
            texts += [f"{USER}{q}\n{BOT}{a}" for q, a in pairs]
        except Exception as e:  # ağ, adres ya da biçim değişikliği: eğitim yine sürer
            stats[name] = f"alınamadı: {type(e).__name__}: {str(e)[:120]}"
            print(f"uyarı: {name} alınamadı: {e}", flush=True)
    return texts, stats
