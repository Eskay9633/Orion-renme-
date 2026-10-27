"""Kişisel/gizli veri temizleyici. Kotlin karşılığı: orion-core/.../privacy/Scrubber.kt."""
from __future__ import annotations

import regex as re  # \p{L} ve değişken uzunluklu lookbehind için

RULES = [
    ("ANAHTAR", re.compile(r"(?:sk-(?:proj-|ant-)?|gsk_|xai-|hf_|AIza|ghp_|gho_|github_pat_|glpat-)[A-Za-z0-9_\-]{16,}")),
    ("ANAHTAR", re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._\-]{12,}")),
    ("PAROLA", re.compile(r"(?i)(?<=(?:şifre|şifrem|sifre|parola|parolam|password|passwd|pin)\s{0,3}[:=]\s{0,3})\S+")),
    ("EPOSTA", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("IBAN", re.compile(r"\bTR\d{2}(?: ?\d{4}){5} ?\d{2}\b")),
    ("KART", re.compile(r"\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}\b")),
    ("TELEFON", re.compile(r"(?:\+90[ \-]?|\b0)5\d{2}[ \-]?\d{3}[ \-]?\d{2}[ \-]?\d{2}\b")),
    ("TCKN", re.compile(r"\b[1-9]\d{10}\b")),
    ("ADRES", re.compile(r"(?i)[\p{L}]+ (?:Mah\.|Mahallesi|Cad\.|Caddesi|Cd\.|Sok\.|Sokak|Sokağı|Sk\.)(?:[ ,]+[\p{L}\d.:/]+){0,6}?[ ,]+No ?: ?\d+[\p{L}]?(?:[ ,]+(?:Daire|D\.|Kat) ?:? ?\d+)?(?: [\p{Lu}][\p{L}]+)?")),
]
HARD = {"ANAHTAR", "PAROLA", "KART", "IBAN", "TCKN"}


def scrub(text: str) -> str:
    for label, rx in RULES:
        text = rx.sub(f"<GIZLI:{label}>", text)
    return text


def contains_secret(text: str) -> bool:
    return any(rx.search(text) for _, rx in RULES)


def contains_hard_secret(text: str) -> bool:
    return any(rx.search(text) for label, rx in RULES if label in HARD)
