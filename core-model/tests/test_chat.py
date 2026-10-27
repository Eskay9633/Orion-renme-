"""Sohbet beyni hattı: küçük yapay metinle uçtan uca (indirme yok)."""
import json

from orion_core import chat


def test_prepare_train_export(tmp_path):
    texts = [f"Orion telefonda yaşayan bir yapay zekadır. Bugün {i}. gün ve hava güzel. İstanbul büyük bir şehirdir." * 3
             for i in range(300)]
    data = tmp_path / "data"
    st = chat.prepare(data, 10**9, vocab=400, texts=texts)
    assert st["train_tokens"] > 1000 and st["val_tokens"] > 0 and st["dialog_tokens"] > 1000
    run = tmp_path / "run"
    tiny = dict(ctx=64, d_model=32, n_layer=1, n_head=2, d_ff=64)
    s = chat.train(data, run, steps=20, budget_min=5, batch=4, lr=3e-3, dialog_frac=0.3, config=tiny)
    assert s["done"] and s["val_loss"] == s["val_loss"]
    s2 = chat.train(data, run, steps=30, budget_min=5, batch=4, lr=3e-3, dialog_frac=0.3, config=tiny)
    assert s2["step"] == 30  # kaldığı yerden devam
    m = chat.export_chat(run, data, tmp_path / "exp")
    assert (tmp_path / "exp" / "model.bin").exists() and m["kind"] == "chat"
    par = json.loads((tmp_path / "exp" / "parity.json").read_text())
    assert par["cases"][0]["prompt"].startswith("Kullanıcı: ")


def test_clean_article_cuts_references():
    t = chat.clean_article("Ankara başkenttir.\n\nTarih\nKuruldu.\nKaynakça\nx y z")
    assert "Kaynakça" not in t and "Kuruldu." in t


def test_persona_dialogues_format():
    d = chat.persona_dialogues(50)
    assert all(x.startswith("Kullanıcı: ") and "\nOrion: " in x for x in d)
