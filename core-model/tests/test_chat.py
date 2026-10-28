"""Sohbet beyni hattı: küçük yapay metinle uçtan uca (indirme yok)."""
import json

from orion_core import chat


def test_prepare_train_export(tmp_path):
    texts = [f"Orion telefonda yaşayan bir yapay zekadır. Bugün {i}. gün ve hava güzel. İstanbul büyük bir şehirdir." * 3
             for i in range(300)]
    data = tmp_path / "data"
    st = chat.prepare(data, 10**9, vocab=400, texts=texts, open_data=False)
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


def test_persona_dialogues_are_diverse_and_question_dependent():
    from orion_core import dialogs
    d = dialogs.persona_dialogues(5000, seed=3)
    answers = {x.split("\nOrion: ")[1] for x in d}
    questions = {x.split("\nOrion: ")[0] for x in d}
    assert len(questions) > 300 and len(answers) > 60
    # Kimlik sorusunun cevabı hep Orion'dan bahseder; yetenek sorusunun cevabı örnek cümle içerir
    for x in d:
        q, a = x[len("Kullanıcı: "):].split("\nOrion: ")
        if q.lower().rstrip("?").endswith("adın ne"):
            assert "Orion" in a
        if "hatırlatıcı kurabilir misin" in q.lower():
            assert "hatırlat" in a.lower()


def test_exam_questions_never_in_training():
    from orion_core import dialogs
    d = dialogs.persona_dialogues(20000, seed=5)
    qs = {x.split("\nOrion: ")[0][len("Kullanıcı: "):].lower().rstrip("?!. ") for x in d}
    assert not (qs & dialogs.TEST_QUESTIONS)


def test_open_data_filter_and_score():
    from orion_core import dialogs
    assert dialogs.clean_pair("Kara delik nedir?", "Kara delik, ışığın bile kaçamadığı bir bölgedir.")
    assert dialogs.clean_pair("Sen kimsin?", "Ben OpenAI tarafından eğitilmiş bir dil modeliyim.") is None
    assert dialogs.clean_pair("What?", "This is an answer written fully in English words only here.") is None
    assert dialogs.clean_pair("Ay nedir?", "Ay, Dünya'nın uydusudur ve aya ilk 1969'da gidildi.")
    good = ["Ben Orion"] * 3 + ["Hatırlatıcı kurarım", "not alırım", "takvime eklerim", "Bunu yapamam", "Günaydın!",
                                "Rica ederim", "Biraz dinlen"]
    assert dialogs.qa_score(good) == 1.0
    assert dialogs.qa_score(["Sıkıldıysan yürüyüş yap"] * 10) < 0.2


def test_clean_reply_strips_leaked_turn_prefixes():
    from orion_core.chat import clean_reply
    assert clean_reply("Orion: Merhaba! Bugün ne yapalım?") == "Merhaba! Bugün ne yapalım?"
    assert clean_reply("Kullanıcı: Merhaba\nOrion: Merhaba! Bugün ne yapalım?") == "Merhaba! Bugün ne yapalım?"
    assert clean_reply("Tamam.\nKullanıcı: başka\nOrion: x") == "Tamam."
    assert clean_reply("Kullanıcı: yalnızca soru") == ""
