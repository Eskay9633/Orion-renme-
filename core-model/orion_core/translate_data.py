"""İngilizce açık lisanslı diyalogları Türkçeye çevirip sohbet beyni için ek veri üretir (GitHub Actions'ta çalışır).

Akış (iş akışı: tools/gelisim/.github/workflows/cevir.yml):
  collect   → 4 veri setini Hugging Face'ten indirir, LİSANS KAPISI'ndan geçirir, (soru, cevap) çiftlerine
              (doğal olduğunda kısa çok turlu diyaloğa) çevirir, süzer, tekilleştirir, N parçaya böler
  translate → bir parçayı CTranslate2 (int8, işlemci) ile en→tr çevirir, Türkçe kalite süzgeçlerinden geçirir
  merge     → parçaları birleştirir, gidiş-dönüş chrF ile en kötü %20'yi atar, istatistik + lisans notu yazar

Kaynaklar (lisanslar ÇALIŞMA ANINDA Hugging Face API'sinden okunur; beklenenle eşleşmezse iş durur):
  OpenAssistant/oasst2               apache-2.0   insan yazımı (yalnız "en" kök sorular)
  databricks/databricks-dolly-15k    cc-by-sa-3.0 insan yazımı; çevirisi de CC BY-SA 3.0 kalır (türev eser)
  Anthropic/hh-rlhf (helpful-base)   mit          Anthropic'in yayınladığı; yalnız "chosen", red-team/harmless YOK
  nvidia/HelpSteer2                  cc-by-4.0    her soru için en yüksek "helpfulness" cevabı
OpenAI üretimi veri kullanılmaz. Çevirmen: Helsinki-NLP/opus-mt-tc-big-en-tr (lisansı da çalışma anında denetlenir);
gidiş-dönüş denetimi: Helsinki-NLP/opus-mt-tc-big-tr-en.

Satır biçimi (çıktı jsonl): {"id","q","a","turns"?,"source","license","translator","translator_license",
"training_allowed": true, "attribution"}. "turns" yalnız çok turlu diyalogda: [[q1,a1],[q2,a2]] (q/a = ilk tur).

Gidiş-dönüş (round-trip) chrF: kimliğin özetine göre diyalogların ~%15'i örneklenir; ilk turun cevabı tr→en geri
çevrilip özgün İngilizceyle chrF (karakter 6-gram, beta=2) hesaplanır. merge, örneklenenlerin chrF dağılımının
alt %20'sini (eşik raporlanır) atar. Örneklenmeyenler bu ölçüyle elenemez; eşiğin altında kalan oran rapordaki
"tahmini kötü oran"dır (kalan veride benzer oranda zayıf çeviri olabileceğinin göstergesi).

Ağ, Hugging Face adresleri, lisans alanları ve CTranslate2 hızı bu depoda DOĞRULANMADI (CI'da denenir).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

HF = "https://huggingface.co"
LICENSE_ALLOW = {"apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0"}
MT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-tr"
BT_MODEL = "Helsinki-NLP/opus-mt-tc-big-tr-en"
MT_LICENSE = "cc-by-4.0"  # beklenen; çalışma anında model kartından doğrulanır

# kaynak adı → depo, beklenen lisans, dosya deseni, ayrıştırıcı adı, atıf
SOURCES: dict[str, dict] = {
    "oasst2": dict(repo="OpenAssistant/oasst2", license="apache-2.0",
                   pattern=r"^data/(train|validation)-.*\.parquet$", parser="oasst2",
                   attribution="OpenAssistant/oasst2 (OpenAssistant katkıcıları, Apache-2.0)"),
    "dolly": dict(repo="databricks/databricks-dolly-15k", license="cc-by-sa-3.0",
                  pattern=r"(^|/)databricks-dolly-15k\.jsonl$", parser="dolly",
                  attribution="databricks/databricks-dolly-15k (© Databricks, Inc., CC BY-SA 3.0); "
                              "Türkçe çevirisi türev eserdir ve CC BY-SA 3.0 ile paylaşılır"),
    "hh-rlhf": dict(repo="Anthropic/hh-rlhf", license="mit",
                    pattern=r"^helpful-base/(train|test)\.jsonl\.gz$", parser="hh",
                    attribution="Anthropic/hh-rlhf helpful-base, yalnız 'chosen' (Anthropic, MIT)"),
    "helpsteer2": dict(repo="nvidia/HelpSteer2", license="cc-by-4.0",
                       pattern=r"^(train|validation)\.jsonl\.gz$", parser="helpsteer2",
                       attribution="nvidia/HelpSteer2 (NVIDIA, CC BY 4.0), en yüksek yararlılık puanlı cevap"),
}


class LicenseError(RuntimeError):
    pass


# --------------------------------------------------------------------------- Hugging Face erişimi

class HfHub:
    """Hugging Face'e düz HTTP erişimi (ek kütüphane gerekmez). Testlerde sahte bir nesneyle değiştirilir."""

    def __init__(self, cache: Path):
        self.cache = Path(cache)

    def _json(self, url: str):
        return json.loads(urllib.request.urlopen(url, timeout=60).read())

    def info(self, repo: str, kind: str = "datasets") -> dict:
        return self._json(f"{HF}/api/{kind}/{repo}")

    def files(self, repo: str) -> list[str]:
        items = self._json(f"{HF}/api/datasets/{repo}/tree/main?recursive=1")
        return sorted(f["path"] for f in items if f.get("type", "file") == "file")

    def download(self, repo: str, path: str) -> Path:
        local = self.cache / repo.replace("/", "__") / path
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            print(f"indiriliyor: {repo}/{path}", flush=True)
            tmp = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(f"{HF}/datasets/{repo}/resolve/main/{path}", tmp)
            tmp.replace(local)
        return local


def license_of(info: dict) -> str | None:
    """API yanıtından lisans: önce cardData.license, yoksa "license:..." etiketi. Küçük harfe çevrilir."""
    lic = (info.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = lic[0] if len(lic) == 1 else None
    if not lic:
        tags = [t.split(":", 1)[1] for t in info.get("tags") or [] if t.startswith("license:")]
        lic = tags[0] if len(tags) == 1 else None
    return str(lic).strip().lower() if lic else None


def check_license(hub, repo: str, expected: str, kind: str = "datasets") -> dict:
    """LİSANS KAPISI: lisans izin listesinde ve beklenenle aynı olmalı; değilse açık Türkçe hatayla durur."""
    try:
        info = hub.info(repo, kind)
    except Exception as e:  # ağ hatası da kapıdan geçemez
        raise LicenseError(f"LİSANS KAPISI: {repo} lisansı okunamadı ({type(e).__name__}: {e}); çeviri durduruldu.")
    lic = license_of(info)
    if lic is None:
        raise LicenseError(f"LİSANS KAPISI: {repo} için lisans bilgisi bulunamadı; çeviri durduruldu.")
    if lic not in LICENSE_ALLOW:
        raise LicenseError(f"LİSANS KAPISI: {repo} lisansı '{lic}' izin listesinde değil "
                           f"({', '.join(sorted(LICENSE_ALLOW))}); çeviri durduruldu.")
    if lic != expected:
        raise LicenseError(f"LİSANS KAPISI: {repo} lisansı '{lic}', beklenen '{expected}'; lisans değişmiş olabilir, "
                           f"elle incelenmeden devam edilmez.")
    return {"repo": repo, "license": lic, "sha": info.get("sha"), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# --------------------------------------------------------------------------- İngilizce süzgeçler

US_STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
             "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
             "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
             "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
             "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas",
             "Utah", "Vermont", "Virginia", "Washington", "Wisconsin", "Wyoming"]  # Georgia: ülke adıyla karışır
US_TERMS = ["Thanksgiving", "SAT", "ACT test", "GPA", "SSN", "Social Security", "401k", "401(k)", "IRA", "IRS", "W-2",
            "Medicare", "Medicaid", "zip code", "Fourth of July", "4th of July", "Super Bowl", "NFL", "NBA", "MLB",
            "Memorial Day", "Labor Day", "Halloween", "DMV", "HOA", "FAFSA", "Walmart", "Costco", "Target store",
            "Fahrenheit", "°F", "mph", "miles", "mile", "feet", "foot", "inches", "inch", "pounds", "lbs", "ounces",
            "oz", "gallon", "gallons", "yard", "yards", "dollars", "cents", "Congress", "Senator", "Republican",
            "Democrat", "Democrats", "Republicans", "Trump", "Obama", "Biden", "the US", "the U.S.", "USA", "America",
            "American", "Americans"]
US_RE = re.compile(r"\$\s?\d|\b\d+\s?°?\s?F\b|(?<![\w])(" + "|".join(re.escape(t) for t in US_STATES + US_TERMS) + r")(?![\w])")
CODE_RE = re.compile(r"```|`[^`\n]+`|\bdef \w+\(|\bfunction\s*\w*\(|\bimport \w+|#include|</?\w+>|\w+\(\)|[{};]\s*$|"
                     r"\\frac|\\sqrt|\$\$|\\\(|\\\[|\b\w+\s*=\s*[\w\d(\[]|\d\s*[*/^]\s*\d|\bSELECT\b.*\bFROM\b|=>|::",
                     re.M)
IDENTITY_EN = re.compile(r"chatgpt|openai|gpt-?[34]|\bclaude\b|anthropic|open ?assistant|language model|\bas an ai\b|"
                         r"\bi(?: am|'m) an ai\b|\bai assistant\b|\bllama\b|\bgemini\b|\bbard\b", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
EN_STOP = {"the", "and", "is", "are", "you", "your", "of", "to", "that", "it", "for", "with", "this", "what", "how",
           "can", "be", "have", "was", "not", "or", "they", "we", "my", "if", "will", "would", "there", "their",
           "which", "about", "some", "like", "but", "just", "does", "do", "should", "could"}
EN_FUNC = EN_STOP | {"a", "an", "in", "on", "at", "i", "me", "to", "it", "is", "am", "by", "as", "from", "who", "why",
                     "when", "where", "his", "her", "he", "she", "its", "our", "us", "them", "every", "all", "no", "yes"}
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.U)
SENT_END = re.compile(r"(?<=[.!?])\s+")


def words(s: str) -> list[str]:
    return WORD_RE.findall(s or "")


def looks_english(s: str) -> bool:
    w = [x.lower() for x in words(s)]
    if not w:
        return False
    letters = [ch for ch in s if ch.isalpha()]
    ascii_ok = sum(ch.isascii() for ch in letters) / max(1, len(letters)) > 0.97  # tırnak/tire Unicode olabilir
    return ascii_ok and (len(w) < 4 or sum(x in EN_FUNC for x in w) / len(w) >= 0.08)


def cut_answer(a: str, soft: int = 80, hard: int = 120) -> str:
    """≤soft sözcükse aynen; değilse cümle sonunda ≤hard sözcüğe kesilir; kesilemezse boş."""
    a = (a or "").strip()
    if len(a.split()) <= soft:
        return a
    out, n = [], 0
    for line in a.split("\n"):
        parts = SENT_END.split(line)
        for i, p in enumerate(parts):
            k = len(p.split())
            if n + k > hard:
                txt = "\n".join(out).strip()
                return txt if txt and txt[-1] in ".!?:" and len(txt.split()) >= 15 else ""
            out_piece = p if i == 0 else " " + p
            if i == 0:
                out.append(out_piece)
            else:
                out[-1] += out_piece
            n += k
    return a


def en_reject(q: str, a: str) -> str | None:
    """İngilizce çifti eleme nedeni (yoksa None)."""
    q, a = (q or "").strip(), (a or "").strip()
    if not (3 <= len(q) <= 500) or len(words(q)) > 70:
        return "soru_uzunluk"
    if len(a) < 2:
        return "cevap_bos"
    if CODE_RE.search(q) or CODE_RE.search(a):
        return "kod_matematik"
    if not looks_english(q) or not looks_english(a):
        return "ingilizce_degil"
    if URL_RE.search(q) or URL_RE.search(a):
        return "adres"
    if IDENTITY_EN.search(q) or IDENTITY_EN.search(a):
        return "model_kimligi"
    if US_RE.search(q) or US_RE.search(a):
        return "abd_ozgu"
    return None


# --------------------------------------------------------------------------- veri setleri → diyaloglar

def _read_jsonl(path: Path) -> Iterable[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_oasst2(rows: list[dict]) -> list[list[tuple[str, str]]]:
    """İngilizce KÖK sorular → en iyi sıradaki cevap; doğal ise (cevaba gelen ilk takip sorusu + en iyi cevabı) 2. tur."""
    rows = [r for r in rows if not r.get("deleted")]
    kids: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("parent_id"):
            kids.setdefault(r["parent_id"], []).append(r)

    def rank(r: dict) -> float:
        v = r.get("rank")
        return 99.0 if v is None or v != v else float(v)

    def best_reply(mid: str) -> dict | None:
        c = [r for r in kids.get(mid, []) if r.get("role") == "assistant" and r.get("lang") == "en"]
        return min(c, key=rank) if c else None

    out = []
    for r in rows:
        if r.get("parent_id") or r.get("role") != "prompter" or r.get("lang") != "en":
            continue
        a1 = best_reply(r["message_id"])
        if a1 is None:
            continue
        turns = [(r.get("text") or "", a1.get("text") or "")]
        for q2 in sorted((k for k in kids.get(a1["message_id"], []) if k.get("role") == "prompter" and k.get("lang") == "en"),
                         key=lambda k: k["message_id"]):
            a2 = best_reply(q2["message_id"])
            if a2 is not None:
                turns.append((q2.get("text") or "", a2.get("text") or ""))
                break
        out.append(turns)
    return out


def parse_dolly(rows: Iterable[dict]) -> list[list[tuple[str, str]]]:
    """Bağlam (context) gerektiren satırlar atılır (özet/çıkarım: soru tek başına anlamsız)."""
    return [[(r.get("instruction") or "", r.get("response") or "")] for r in rows if not (r.get("context") or "").strip()]


HH_SPLIT = re.compile(r"\n\nHuman: |\n\nAssistant: ")


def parse_hh(rows: Iterable[dict], max_turns: int = 2) -> list[list[tuple[str, str]]]:
    """helpful-base "chosen" metni: "\\n\\nHuman: ...\\n\\nAssistant: ..." → turlar (en çok max_turns)."""
    out = []
    for r in rows:
        text = r.get("chosen") or ""
        roles = re.findall(r"\n\n(Human|Assistant): ", text)
        parts = HH_SPLIT.split(text)[1:]
        if len(roles) != len(parts) or len(parts) < 2:
            continue
        turns = []
        for i in range(0, len(parts) - 1, 2):
            if roles[i] != "Human" or roles[i + 1] != "Assistant":
                break
            turns.append((parts[i], parts[i + 1]))
        if turns and len(turns) <= max_turns:
            out.append(turns)
    return out


def parse_helpsteer2(rows: Iterable[dict]) -> list[list[tuple[str, str]]]:
    """Her soru için en yüksek helpfulness (eşitlikte correctness, sonra kısa olan) cevabı; çok turlu istemler atılır."""
    best: dict[str, dict] = {}

    def key(r):
        return (r.get("helpfulness") or 0, r.get("correctness") or 0, -len(r.get("response") or ""))

    for r in rows:
        p = r.get("prompt") or ""
        if "<extra_id" in p:
            continue
        if p not in best or key(r) > key(best[p]):
            best[p] = r
    return [[(p, r.get("response") or "")] for p, r in best.items()]


PARSERS = {"oasst2": parse_oasst2, "dolly": parse_dolly, "hh": parse_hh, "helpsteer2": parse_helpsteer2}


def load_source(hub, name: str) -> list[list[tuple[str, str]]]:
    spec = SOURCES[name]
    pat = re.compile(spec["pattern"])
    paths = [p for p in hub.files(spec["repo"]) if pat.search(p)]
    if not paths:
        raise RuntimeError(f"{spec['repo']}: beklenen dosya bulunamadı (desen {spec['pattern']})")
    rows: list[dict] = []
    for p in paths:
        local = hub.download(spec["repo"], p)
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            rows += pq.read_table(local).to_pylist()
        else:
            rows += list(_read_jsonl(local))
    return PARSERS[spec["parser"]](rows)


def clean_dialog(turns: list[tuple[str, str]], stats: Counter) -> list[tuple[str, str]]:
    """Turları süzer; ilk tur geçemezse diyalog düşer, sonraki tur geçemezse oradan kesilir."""
    kept = []
    for q, a in turns:
        a2 = cut_answer(a)
        why = "cevap_uzun" if not a2 else en_reject(q, a2)
        if why:
            if not kept:
                stats[why] += 1
            else:
                stats["tur_kesildi"] += 1
            break
        kept.append((q.strip(), a2))
    return kept


def norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def collect(sources: list[str], limit: int, out: Path, n_shards: int = 3, hub=None, seed: int = 1) -> dict:
    """İndirir + lisans kapısı + süzgeç + tekilleştirme + parçalama. out/shard-XX.jsonl, out/collect.json yazar."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hub = hub or HfHub(out / "hf")
    # Önce TÜM lisanslar (veri seti + çevirmen + geri çevirmen); biri bile uymazsa hiçbir şey indirilmez
    licenses = {name: check_license(hub, SOURCES[name]["repo"], SOURCES[name]["license"]) for name in sources}
    licenses["translator"] = check_license(hub, MT_MODEL, MT_LICENSE, kind="models")
    licenses["backtranslator"] = check_license(hub, BT_MODEL, MT_LICENSE, kind="models")
    stats: dict = {"kaynak": {}, "atilan": {}}
    seen_pair, seen_q = set(), set()
    dialogs = []
    for name in sources:
        drop: Counter = Counter()
        try:
            raw = load_source(hub, name)
        except Exception as e:  # adres/biçim değişikliği: bu kaynak atlanır, raporlanır (lisans hatası değil)
            print(f"uyarı: {name} alınamadı: {e}", flush=True)
            stats["kaynak"][name] = f"alınamadı: {type(e).__name__}: {str(e)[:160]}"
            continue
        kept = 0
        for turns in raw:
            t = clean_dialog(turns, drop)
            if not t:
                continue
            k1, kp = norm_key(t[0][0]), norm_key(t[0][0] + " || " + t[0][1])
            if k1 in seen_q or kp in seen_pair:
                drop["tekrar"] += 1
                continue
            seen_q.add(k1)
            seen_pair.add(kp)
            did = name + "-" + hashlib.sha1(kp.encode()).hexdigest()[:12]
            dialogs.append({"id": did, "source": SOURCES[name]["repo"], "license": licenses[name]["license"],
                            "attribution": SOURCES[name]["attribution"], "turns": [list(x) for x in t]})
            kept += 1
        stats["kaynak"][name] = {"ham": len(raw), "kalan": kept}
        stats["atilan"][name] = dict(drop)
    if not dialogs:
        raise RuntimeError("hiçbir kaynaktan diyalog alınamadı: " + json.dumps(stats["kaynak"], ensure_ascii=False))
    rng = random.Random(seed)
    rng.shuffle(dialogs)
    if limit and len(dialogs) > limit:
        dialogs = dialogs[:limit]
    shards = [[] for _ in range(max(1, n_shards))]
    for i, d in enumerate(dialogs):
        shards[i % len(shards)].append(d)
    for i, s in enumerate(shards):
        with open(out / f"shard-{i:02d}.jsonl", "w", encoding="utf-8") as f:
            for d in s:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats.update({"diyalog": len(dialogs), "tur": sum(len(d["turns"]) for d in dialogs), "parca": len(shards),
                  "lisanslar": licenses, "limit": limit})
    (out / "licenses.json").write_text(json.dumps(licenses, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "collect.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


# --------------------------------------------------------------------------- Türkçe kalite süzgeçleri

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_STOP = {"ve", "bir", "bu", "için", "ile", "da", "de", "mi", "mı", "ne", "çok", "daha", "gibi", "olarak", "ama",
           "veya", "en", "olan", "ki", "her", "şey", "sen", "ben", "o", "var", "yok", "nasıl", "neden", "ya"}
SIZ_WORDS = {"siz": "sen", "sizin": "senin", "size": "sana", "sizi": "seni", "sizde": "sende", "sizden": "senden",
             "sizce": "sence", "sizinle": "seninle", "sizinkini": "seninkini", "sizinki": "seninki"}
# 2. çoğul kişi eki (-nız/-niz/-sınız/-iniz...) → tekil (-n/-sın/-in): "yapabilirsiniz"→"yapabilirsin",
# "bilgisayarınızı"→"bilgisayarını", "evinizde"→"evinde". Kök ≥3 harf; "deniz" gibi sözcükler hariç.
SIZ_SUFFIX = re.compile(r"\b(\w{2,}[aeıioöuü])n[ıiuü]z(ı|i|u|ü|a|e|da|de|ta|te|dan|den|tan|ten|la|le|dır|dir|dur|dür)?\b")
SIZ_EXCEPT = ("deniz", "yalnız", "beniz", "anız")
SIZ_LEFT = re.compile(r"\bsiz(in|e|i|de|den|ce|inle)?\b|(sınız|siniz|sunuz|sünüz)\b", re.I)
SEN_MARK = re.compile(r"\b(sen|senin|sana|seni|sende|senden|sence|seninle)\b|"
                      r"(abilir|ebilir|yor|malı|meli|acak|ecek)(sın|sin|sun|sün)\b", re.I)


def to_sen(text: str) -> str:
    """Orion "sen" diye hitap eder: belirgin "siz" biçimlerini tekile çevirir (sezgisel, kesin değil).
    Çoğul emir kipi ("deneyin") belirsiz olduğu için (ilgi hâli eki ile aynı) değiştirilmez."""
    def w(m):
        s = m.group(0)
        r = SIZ_WORDS[s.lower()]
        return r.capitalize() if s[0].isupper() else r

    text = re.sub(r"\b(" + "|".join(SIZ_WORDS) + r")\b", w, text, flags=re.I)

    def suf(m):
        whole = m.group(0)
        if any(x in whole.lower() for x in SIZ_EXCEPT):
            return whole
        return m.group(1) + "n" + (m.group(2) or "")

    return SIZ_SUFFIX.sub(suf, text)


def tr_ratio(s: str) -> float:
    w = [x.lower() for x in words(s)]
    if not w:
        return 0.0
    return sum(x in TR_STOP or any(c in TR_CHARS for c in x) for x in w) / len(w)


def untranslated_ratio(src: str, tgt: str) -> float:
    """Çeviride kalan İngilizce: İngilizce işlev sözcükleri + kaynaktan aynen kopyalanmış uzun sözcükler oranı."""
    tw = [x.lower() for x in words(tgt)]
    if not tw:
        return 1.0
    sw = {x.lower() for x in words(src) if len(x) >= 4}
    stop = sum(x in EN_STOP for x in tw)
    copied = sum(len(x) >= 4 and x.isascii() and x in sw for x in tw)
    return max(stop / len(tw), copied / len(tw) * 0.5)  # özel adlar kopyalanır: kopya yarım ağırlıklı


def has_loop(s: str) -> bool:
    w = [x.lower() for x in words(s)]
    for i in range(len(w) - 3):
        if w[i] == w[i + 1] == w[i + 2] == w[i + 3]:
            return True
    tri = Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    if not tri:
        return False
    top = tri.most_common(1)[0][1]
    return top >= 4 or (top >= 3 and len(set(w)) / len(w) < 0.5)


def tr_reject(src: str, tgt: str) -> str | None:
    tgt = (tgt or "").strip()
    if not tgt:
        return "ceviri_bos"
    if len(src) >= 20:
        r = len(tgt) / len(src)
        if not (0.7 <= r <= 1.5):
            return "uzunluk_orani"
    if has_loop(tgt):
        return "tekrar_dongusu"
    n = len(words(tgt))
    if n >= 4 and untranslated_ratio(src, tgt) > 0.12:
        return "cevrilmemis_ingilizce"
    if n >= 5 and tr_ratio(tgt) < 0.15:
        return "turkce_degil"
    return None


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015): karakter n-gram F-beta, boşluklar atılarak; 0..100."""
    h, r = re.sub(r"\s+", "", hyp or ""), re.sub(r"\s+", "", ref or "")
    precs, recs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        m = sum((hc & rc).values())
        precs.append(m / sum(hc.values()))
        recs.append(m / sum(rc.values()))
    if not precs:
        return 0.0
    p, rr = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rr == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


def sampled(did: str, pct: int = 15) -> bool:
    return int(hashlib.sha1(did.encode()).hexdigest()[:8], 16) % 100 < pct


# --------------------------------------------------------------------------- çeviri

LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")


def segments(text: str) -> list[tuple[str, list[str]]]:
    """Metni satır → (liste öneki, cümleler) olarak böler; Marian modelleri cümle düzeyinde daha iyi çevirir."""
    out = []
    for line in text.split("\n"):
        m = LIST_PREFIX.match(line)
        pre = m.group(1) if m else ""
        body = line[len(pre):].strip()
        out.append((pre, [s for s in SENT_END.split(body) if s.strip()] if body else []))
    return out


def translate_texts(texts: list[str], fn: Callable[[list[str]], list[str]]) -> list[str]:
    """Metinleri cümlelere bölüp tek toplu çağrıyla çevirir, satır/liste düzenini koruyarak geri birleştirir."""
    segs = [segments(t) for t in texts]
    uniq = sorted({s for sg in segs for _, ss in sg for s in ss})
    tr = dict(zip(uniq, fn(uniq))) if uniq else {}
    res = []
    for sg in segs:
        lines = [(pre + " ".join(tr[s].strip() for s in ss)) if ss else pre.rstrip() for pre, ss in sg]
        res.append("\n".join(lines).strip())
    return res


class Ct2Translator:
    """CTranslate2 (int8, işlemci) + Hugging Face tokenizer. Modeli önce dönüştür:
    ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-en-tr --output_dir ct2-en-tr --quantization int8"""

    def __init__(self, ct2_dir: str, hf_name: str, threads: int = 4, beam: int = 2, batch: int = 32, prefix: str = ""):
        import ctranslate2
        import transformers
        self.tok = transformers.AutoTokenizer.from_pretrained(hf_name)
        self.tr = ctranslate2.Translator(ct2_dir, device="cpu", compute_type="int8", intra_threads=threads)
        self.beam, self.batch, self.prefix = beam, batch, prefix

    def __call__(self, texts: list[str]) -> list[str]:
        src = [self.tok.convert_ids_to_tokens(self.tok.encode(self.prefix + t)) for t in texts]
        res = self.tr.translate_batch(src, max_batch_size=self.batch, beam_size=self.beam,
                                      max_decoding_length=256, repetition_penalty=1.1)
        return [self.tok.decode(self.tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res]


def translate_shard(shard: Path, out: Path, translator: Callable, backtranslator: Callable | None = None,
                    licenses: dict | None = None, budget_min: float = 0, chunk: int = 64, sample_pct: int = 15) -> dict:
    """Bir parçayı çevirir; çıktı out (jsonl) ve out.stats.json. Süre bütçesi dolarsa o ana kadarki kısım kalır."""
    licenses = licenses or {}
    tl = (licenses.get("translator") or {}).get("license", MT_LICENSE)
    rows = list(_read_jsonl(shard))
    drop: Counter = Counter()
    t0 = time.time()
    kept = done = 0
    chrfs = []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), chunk):
            if budget_min and time.time() - t0 > budget_min * 60:
                print("süre bütçesi doldu; kalan diyaloglar çevrilmedi", flush=True)
                break
            part = rows[i:i + chunk]
            flat = [x for d in part for t in d["turns"] for x in t]
            tr = translate_texts(flat, translator)
            k = 0
            results = []
            for d in part:
                tt = []
                for q, a in d["turns"]:
                    tt.append((q, a, tr[k], tr[k + 1]))
                    k += 2
                results.append((d, tt))
            bt_items = [(d, tt) for d, tt in results if backtranslator and sampled(d["id"], sample_pct)]
            bt = translate_texts([tt[0][3] for _, tt in bt_items], backtranslator) if bt_items else []
            bt_score = {d["id"]: chrf(b, tt[0][1]) for (d, tt), b in zip(bt_items, bt)}
            for d, tt in results:
                done += 1
                turns = []
                for q, a, tq, ta in tt:
                    tq, ta = to_sen(tq), to_sen(ta)
                    why = tr_reject(q, tq) or tr_reject(a, ta)
                    if why:
                        if not turns:
                            drop[why] += 1
                        else:
                            drop["tur_kesildi"] += 1
                        break
                    turns.append([tq, ta])
                if not turns:
                    continue
                if any(SIZ_LEFT.search(x) for t in turns for x in t) and any(SEN_MARK.search(x) for t in turns for x in t):
                    drop["sen_siz_karisik"] += 1
                    continue
                row = {"id": d["id"], "q": turns[0][0], "a": turns[0][1]}
                if len(turns) > 1:
                    row["turns"] = turns
                row.update({"source": d["source"], "license": d["license"], "translator": f"{MT_MODEL} (CTranslate2 int8)",
                            "translator_license": tl, "training_allowed": True,
                            "attribution": d["attribution"] + f"; makine çevirisi: {MT_MODEL} ({tl})"})
                if d["id"] in bt_score:
                    row["_chrf"] = round(bt_score[d["id"]], 2)
                    chrfs.append(row["_chrf"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            el = time.time() - t0
            print(f"{done}/{len(rows)} diyalog, {kept} kaldı, {done / max(el, 1e-9):.2f} diyalog/sn", flush=True)
    el = time.time() - t0
    st = {"parca": shard.name, "girdi": len(rows), "islenen": done, "kalan": kept, "atilan": dict(drop),
          "yarim": done < len(rows), "sure_sn": round(el), "hiz_diyalog_sn": round(done / max(el, 1e-9), 3),
          "chrf_ornek": len(chrfs)}
    Path(str(out) + ".stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return st


# --------------------------------------------------------------------------- birleştirme

def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = (len(s) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


NOTES = """# Orion çeviri verisi — lisans ve atıf

Bu dosyadaki her satır İngilizce açık lisanslı bir veri setinden makine çevirisiyle (en→tr) üretilmiştir.
Her satırın `source`, `license`, `translator`, `attribution` alanları kaynağını gösterir.

- OpenAssistant/oasst2 — Apache-2.0 (OpenAssistant katkıcıları)
- databricks/databricks-dolly-15k — CC BY-SA 3.0 (© Databricks, Inc.). Bu satırların çevirisi türev eserdir ve
  **CC BY-SA 3.0** ile paylaşılır (aynı lisansla paylaş şartı).
- Anthropic/hh-rlhf (helpful-base, yalnız "chosen") — MIT (Anthropic)
- nvidia/HelpSteer2 — CC BY 4.0 (NVIDIA)
- Çevirmen model: Helsinki-NLP/opus-mt-tc-big-en-tr — lisansı çalışma anında model kartından okundu (aşağıda).
  Makine çevirisi çıktısının ayrıca bir lisans koşulu olup olmadığı: doğrulanmadı.

OpenAI ya da başka kapalı model çıktısı kullanılmamıştır. Lisanslar yayın anında Hugging Face API'sinden okunmuştur;
hukuki değerlendirme yapılmamıştır (doğrulanmadı). Eğitilmiş modele CC BY-SA şartının geçip geçmediği: doğrulanmadı.
"""


def merge(parts: list[Path], out_dir: Path, drop_frac: float = 0.2, collect_stats: Path | None = None) -> dict:
    """Parçaları birleştirir; örneklenen chrF'nin alt drop_frac dilimini atar; ceviri.jsonl.gz + stats.json + LISANS.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, shard_stats = [], []
    for p in sorted(parts, key=str):
        rows += list(_read_jsonl(p))
        sp = Path(str(p) + ".stats.json")
        if sp.exists():
            shard_stats.append(json.loads(sp.read_text(encoding="utf-8")))
    scores = [r["_chrf"] for r in rows if "_chrf" in r]
    thr = percentile(scores, drop_frac) if scores else None
    drop: Counter = Counter()
    for st in shard_stats:
        drop.update(st.get("atilan", {}))
    kept, seen = [], set()
    for r in rows:
        c = r.pop("_chrf", None)
        if c is not None and thr is not None and c < thr:
            drop["gidis_donus_chrf"] += 1
            continue
        if r["id"] in seen:
            drop["tekrar"] += 1
            continue
        seen.add(r["id"])
        kept.append(r)
    by_src = Counter(r["source"] for r in kept)
    chars = sum(len(t) for r in kept for pair in (r.get("turns") or [[r["q"], r["a"]]]) for t in pair)
    with gzip.open(out_dir / "ceviri.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lic = {}
    if collect_stats and Path(collect_stats).exists():
        cs = json.loads(Path(collect_stats).read_text(encoding="utf-8"))
        lic = cs.get("lisanslar", {})
    else:
        cs = {}
    stats = {"kalan": len(kept), "cok_turlu": sum("turns" in r for r in kept), "kaynak": dict(by_src),
             "atilan": dict(drop), "chrf": {"ornek": len(scores), "esik_p20": thr,
                                            "ortalama": round(sum(scores) / len(scores), 2) if scores else None,
                                            "medyan": percentile(scores, 0.5) if scores else None},
             "karakter": chars, "tahmini_belirtec": chars // 4,  # Orion BPE'si ~4 karakter/belirteç (tahmin)
             "parcalar": shard_stats, "yarim_parca": [s["parca"] for s in shard_stats if s.get("yarim")],
             "toplama": {k: cs.get(k) for k in ("kaynak", "atilan", "diyalog", "limit")} if cs else {},
             "lisanslar": lic}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    notes = NOTES + "\n## Çalışma anında okunan lisanslar\n\n" + "\n".join(
        f"- {v.get('repo')}: {v.get('license')} (sha {v.get('sha')}, {v.get('checked_at')})" for v in lic.values()) + "\n"
    (out_dir / "LISANS-ATIF.md").write_text(notes, encoding="utf-8")
    return stats


# --------------------------------------------------------------------------- komut satırı

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="İngilizce açık lisanslı diyalogları Türkçeye çevirir")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--pairs", type=int, default=100_000)
    c.add_argument("--shards", type=int, default=3)
    c.add_argument("--sources", default=",".join(SOURCES))
    t = sub.add_parser("translate")
    t.add_argument("--shard", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--ct2", required=True, help="dönüştürülmüş en→tr CTranslate2 klasörü")
    t.add_argument("--ct2-back", help="dönüştürülmüş tr→en klasörü (gidiş-dönüş denetimi)")
    t.add_argument("--licenses", help="collect'in yazdığı licenses.json")
    t.add_argument("--budget-min", type=float, default=320)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--beam", type=int, default=2)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.add_argument("--collect", help="collect.json (lisans kaydı için)")
    m.add_argument("parts", nargs="+")
    lc = sub.add_parser("license-check", help="yalnız çevirmen modellerinin lisansını denetler")
    lc.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "collect":
            st = collect([s for s in a.sources.split(",") if s], a.pairs, Path(a.out), a.shards)
            st = {k: v for k, v in st.items() if k != "lisanslar"}
        elif a.cmd == "translate":
            hub = HfHub(Path("."))
            lic = json.loads(Path(a.licenses).read_text(encoding="utf-8")) if a.licenses else {}
            lic["translator"] = check_license(hub, MT_MODEL, MT_LICENSE, kind="models")  # çevirmeden hemen önce yeniden
            tr = Ct2Translator(a.ct2, MT_MODEL, threads=a.threads, beam=a.beam)
            bt = Ct2Translator(a.ct2_back, BT_MODEL, threads=a.threads, beam=1) if a.ct2_back else None
            st = translate_shard(Path(a.shard), Path(a.out), tr, bt, lic, a.budget_min)
        elif a.cmd == "merge":
            st = merge([Path(p) for p in a.parts], Path(a.out), collect_stats=Path(a.collect) if a.collect else None)
            st = {k: v for k, v in st.items() if k != "parcalar"}
        else:
            hub = HfHub(Path("."))
            st = {"translator": check_license(hub, MT_MODEL, MT_LICENSE, kind="models"),
                  "backtranslator": check_license(hub, BT_MODEL, MT_LICENSE, kind="models")}
            if a.out:
                Path(a.out).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    except LicenseError as e:
        print(f"HATA: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
    print(json.dumps(st, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
