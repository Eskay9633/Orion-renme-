"""Mastar işaretçisi '+k' çözümü: yalnızca -me/-ma ile biten aralığa 'k' eklenir.
Aynı örnekler Kotlin'de: orion-core/src/test/kotlin/dev/orion/core/CopyDecodeTest.kt (eşlik)."""
from orion_core import SHARED
from orion_core.bpe import BPE
from orion_core.copy import decode_output, find_span

# (girdi, kopyalanan aralık, '+k' var mı, beklenen değer) — Kotlin testindeki CASES ile aynı
CASES = [
    ("faturayı ödemeyi hatırlat", "faturayı ödeme", True, "faturayı ödemek"),
    ("kitap okumayı hatırlat", "kitap okuma", True, "kitap okumak"),
    ("markete gitmeden önce hatırlat", "markete", True, "markete"),
    ("ilaç al hatırlat", "ilaç", True, "ilaç"),
    ("çayı demlemeyi unutma", "çayı demlemeyi", True, "çayı demlemeyi"),
    ("not al: Kayak", "Kayak", False, "Kayak"),
    ("not al: süt al", "süt al", False, "süt al"),
]


def _ids(bpe, src_ids, span, k):
    i, j = find_span(bpe, src_ids, span)
    base = bpe.vocab_size
    return bpe.encode('{"text":"') + [base + i, base + j] + bpe.encode(('k' if k else '') + '"}')


def test_infinitive_k_only_after_me_ma():
    bpe = BPE.load(SHARED / "core-model" / "tokenizer.json")
    for src, span, k, want in CASES:
        src_ids = bpe.encode(src)
        assert decode_output(bpe, src_ids, _ids(bpe, src_ids, span, k)) == '{"text":"%s"}' % want, src
