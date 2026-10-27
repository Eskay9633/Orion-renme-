import json

from orion_core import pipeline
from orion_core.train import load_distilled


def rec(**kw):
    base = {"task": "t2c", "src": "not al: süt", "tgt": '{"cmd":"note.write","args":{"text":"süt"}}',
            "provider": "groq", "model": "m", "date": "2026-10-26", "license": "x",
            "training_allowed": True, "origin": "api", "user_rating": 0}
    base.update(kw)
    return base


P = pipeline.provider_defaults()


def test_manual_import_never_trainable():
    r = pipeline.clean_record(rec(provider="arena", origin="manual_import", training_allowed=True), P)
    assert r["training_allowed"] is False
    r = pipeline.clean_record(rec(provider="verdent", origin="api", training_allowed=True), P)
    assert r["training_allowed"] is False


def test_unverified_teacher_rows_never_trainable():
    assert pipeline.clean_record(rec(origin="teacher"), P)["training_allowed"] is False
    assert pipeline.clean_record(rec(origin="teacher", verified="true"), P)["training_allowed"] is False
    assert pipeline.clean_record(rec(origin="teacher", verified=True), P)["training_allowed"] is True


def test_non_true_values_become_false():
    for v in ("true", 1, None, "yes"):
        assert pipeline.clean_record(rec(training_allowed=v), P)["training_allowed"] is False


def test_hard_secrets_dropped_and_pii_masked():
    assert pipeline.clean_record(rec(src="şifrem: abc123 not al"), P) is None
    assert pipeline.clean_record(rec(src="anahtar sk-proj-abcdefghijklmnopqrstuvwx1234"), P) is None
    r = pipeline.clean_record(rec(src="0532 123 45 67 numarasını not al"), P)
    assert "<GIZLI:TELEFON>" in r["src"]


def test_invalid_command_targets_dropped_and_canonicalized():
    assert pipeline.clean_record(rec(tgt='{"cmd":"shell.exec","args":{}}'), P) is None
    assert pipeline.clean_record(rec(tgt="bozuk"), P) is None
    r = pipeline.clean_record(rec(tgt='{"cmd": "note.write", "args": {"text": "süt"}}'), P)
    assert r["tgt"] == '{"args":{"text":"süt"},"cmd":"note.write"}'


def test_dedupe_ignores_case_and_spaces():
    rows = [rec(src="Not al:  süt"), rec(src="not al: süt"), rec(src="not al: ekmek")]
    assert len(pipeline.dedupe(rows)) == 2


def test_training_filter_end_to_end(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [rec(), rec(src="not al: ekmek", training_allowed=False), rec(src="x", provider="arena", origin="manual_import")]
    raw.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats = pipeline.process([raw], tmp_path / "out")
    assert stats["unique"] == 3 and stats["trainable"] == 1
    train = pipeline.load_jsonl(tmp_path / "out" / "train.jsonl")
    assert all(r["training_allowed"] is True for r in train)
    # Eğitim yükleyicisi de ikinci kez filtreler
    all_path = tmp_path / "out" / "all.jsonl"
    assert len(load_distilled([str(all_path)])) == 1


def test_license_notes_generation(tmp_path):
    p = tmp_path / "LN.md"
    pipeline.write_license_notes(p)
    text = p.read_text(encoding="utf-8")
    assert "Verdent" in text and "Arena" in text and "doğrulanmadı" in text


def test_distill_only_uses_trainable_auto_providers(monkeypatch):
    from orion_core import distill
    for k in list(__import__("os").environ):
        if k.startswith("ORION_KEY_"):
            monkeypatch.delenv(k)
    for pid in ("groq", "gemini", "arena", "verdent", "claude", "openrouter"):
        monkeypatch.setenv(f"ORION_KEY_{pid.upper()}", "x")
    ids = {p["id"] for p in distill.eligible(pipeline.provider_defaults())}
    assert ids == {"groq", "openrouter"}


def test_distill_keep_line_requires_verbatim_values():
    from orion_core.distill import keep_line
    assert keep_line("Yarın sabah 9'da toplantı diye hatırlat", {"text": "toplantı", "at": "d+1@09:00"})
    assert not keep_line("Yarın sabah 9'da görüşme hatırlat", {"text": "toplantı"})
    assert not keep_line('{"cmd":"x"}', {})
