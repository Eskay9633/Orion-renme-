"""İçe aktarma kararları (ağ yok)."""
from orion_core.importer import chat_ok, komut_better

FLOOR = {"val_ppl_max": 60, "persona_ok": True}


def test_komut_regression_rejected():
    cur = {"sha256": "a", "eval": {"handwritten_exact": 0.96, "t2c_ood_exact": 0.67}}
    assert komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.97, "t2c_ood_exact": 0.70}}, cur)[0]
    assert not komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.93, "t2c_ood_exact": 0.80}}, cur)[0]
    assert not komut_better({"sha256": "a", "eval": {}}, cur)[0]
    assert komut_better({"sha256": "b", "eval": {}}, None)[0]


def test_chat_floor():
    assert not chat_ok({"val_ppl_int8": 120, "persona_ok": True, "sha256": "x"}, FLOOR, None)[0]
    assert not chat_ok({"val_ppl_int8": 40, "persona_ok": False, "sha256": "x"}, FLOOR, None)[0]
    assert chat_ok({"val_ppl_int8": 40, "persona_ok": True, "sha256": "x"}, FLOOR, None)[0]
    assert not chat_ok({"val_ppl_int8": 45, "persona_ok": True, "sha256": "y"}, FLOOR, {"val_ppl_int8": 40, "sha256": "x"})[0]


def test_chat_exam_floor():
    """Sınav sorularına soruya bağlı doğru cevap vermeyen (ezber) sohbet beyni alınmaz; gerçek taban dosyası da bunu ister."""
    import json
    from orion_core import SHARED
    real = json.loads((SHARED / "chat-model-floor.json").read_text(encoding="utf-8"))
    assert real["qa_min"] >= 0.7
    floor = dict(FLOOR, qa_min=0.7)
    assert not chat_ok({"val_ppl_int8": 14, "persona_ok": True, "qa_score": 0.3, "sha256": "x"}, floor, None)[0]
    assert not chat_ok({"val_ppl_int8": 14, "persona_ok": True, "sha256": "x"}, floor, None)[0]
    assert chat_ok({"val_ppl_int8": 14, "persona_ok": True, "qa_score": 0.8, "sha256": "x"}, floor, None)[0]


def test_chat_picks_newest_passing_candidate():
    """Sonradan gelen zayıf model (ör. CPU hattı, sınavsız) iyi Kaggle modelinin alınmasını engellemez."""
    from orion_core.importer import pick_chat
    floor = dict(FLOOR, qa_min=0.7)
    weak = ("sohbet-0020000", {"val_ppl_int8": 30, "persona_ok": True, "sha256": "a"})
    good = ("sohbet-kaggle-orta-0070000", {"val_ppl_int8": 12, "persona_ok": True, "qa_score": 0.8, "sha256": "b"})
    i, tried = pick_chat([weak, good], floor, None)
    assert i == 1 and not tried[0]["alındı"] and tried[1]["alındı"]
    assert pick_chat([weak], floor, None)[0] is None


# ---- Yerel yeniden değerlendirme ve bekletme (ağ yok: indirme ve evaluate_release taklit edilir) ----

def _fake_world(monkeypatch, tmp_path, manifests: dict, local: dict, hold: bool):
    """manifests: tag → manifest (en yeniden eskiye); local: tag → evaluate_release sonucu."""
    import json
    from orion_core import importer
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    floor = {"val_ppl_max": 60, "persona_ok": True, "qa_min": 0.7}
    if hold:
        floor["hold"] = True
    (shared / "chat-model-floor.json").write_text(json.dumps(floor), encoding="utf-8")
    monkeypatch.setattr(importer, "SHARED", shared)
    monkeypatch.setattr(importer, "latest", lambda prefix, token: None)
    rels = [{"tag_name": t, "created_at": f"2026-10-{20 - i:02d}", "assets": []} for i, t in enumerate(manifests)]
    monkeypatch.setattr(importer, "recent", lambda prefix, token, n=8: rels)
    downloads: list[tuple[str, tuple]] = []

    def fake_download(rel, names, dest, token):
        downloads.append((rel["tag_name"], tuple(names)))
        for n in names:
            if n == "manifest.json":
                (dest / n).write_text(json.dumps(manifests[rel["tag_name"]]), encoding="utf-8")
            else:
                (dest / n).write_text(f"{rel['tag_name']}:{n}", encoding="utf-8")

    monkeypatch.setattr(importer, "download", fake_download)
    evaluated: list[str] = []

    def fake_eval(d):
        evaluated.append(d.name)
        assert (d / "model.bin").exists() and (d / "tokenizer.json").exists()
        return local[d.name]

    monkeypatch.setattr(importer, "_local_eval", fake_eval)
    return importer, shared, downloads, evaluated


def test_local_scores_override_manifest(monkeypatch, tmp_path):
    """Manifest 0.9 dese de yerel sınav 0.3 ise alınmaz; manifest düşük dese de yerel 0.8 ise alınır."""
    man = {"sohbet-b": {"val_ppl_int8": 12, "persona_ok": True, "qa_score": 0.9, "sha256": "b"},
           "sohbet-a": {"val_ppl_int8": 14, "persona_ok": True, "qa_score": 0.3, "sha256": "a"}}
    loc = {"sohbet-b": {"qa_score": 0.3, "persona_ok": True, "answers": []},
           "sohbet-a": {"qa_score": 0.8, "persona_ok": True, "answers": []}}
    importer, shared, _, evaluated = _fake_world(monkeypatch, tmp_path, man, loc, hold=False)
    rep = importer.run(None, lambda d: True)
    assert evaluated == ["sohbet-b", "sohbet-a"]
    b, a = rep["sohbet_adaylar"]
    assert not b["alındı"] and b["yerel_qa"] == 0.3 and b["manifest_qa"] == 0.9
    assert a["alındı"] and a["yerel_qa"] == 0.8 and a["manifest_qa"] == 0.3
    assert rep["changed"] and rep["sohbet"]["tag"] == "sohbet-a"
    assert (shared / "chat-model" / "model.bin").read_text() == "sohbet-a:model.bin"


def test_hold_prevents_import(monkeypatch, tmp_path):
    man = {"sohbet-a": {"val_ppl_int8": 12, "persona_ok": True, "qa_score": 0.8, "sha256": "a"}}
    loc = {"sohbet-a": {"qa_score": 0.75, "persona_ok": True, "answers": []}}
    importer, shared, _, evaluated = _fake_world(monkeypatch, tmp_path, man, loc, hold=True)
    rep = importer.run(None, lambda d: True)
    assert evaluated == ["sohbet-a"]
    assert rep["changed"] is False
    s = rep["sohbet"]
    assert s["alındı"] is False and s["neden"] == "beklemede (kullanıcı onayı)" and s["yerel_qa"] == 0.75
    assert not (shared / "chat-model").exists()


def test_real_floor_is_on_hold():
    """Kullanıcı onayı gelene kadar sohbet beyni uygulamaya alınmaz (yeni APK yok)."""
    import json
    from orion_core import SHARED
    assert json.loads((SHARED / "chat-model-floor.json").read_text(encoding="utf-8")).get("hold") is True


def test_newest_passing_and_download_limit(monkeypatch, tmp_path):
    """Ucuz şaşkınlık denetimini geçemeyen indirilmez; en çok 3 aday indirilip değerlendirilir; geçen en yeni seçilir."""
    man = {"sohbet-e": {"val_ppl_int8": 300, "persona_ok": True, "sha256": "e"},
           "sohbet-d": {"val_ppl_int8": 20, "persona_ok": True, "sha256": "d"},
           "sohbet-c": {"val_ppl_int8": 20, "persona_ok": True, "sha256": "c"},
           "sohbet-b": {"val_ppl_int8": 20, "persona_ok": True, "sha256": "b"},
           "sohbet-a": {"val_ppl_int8": 20, "persona_ok": True, "sha256": "a"}}
    loc = {"sohbet-d": {"qa_score": 0.8, "persona_ok": True}, "sohbet-c": {"qa_score": 0.9, "persona_ok": True}}
    importer, _, downloads, evaluated = _fake_world(monkeypatch, tmp_path, man, loc, hold=False)
    rep = importer.run(None, lambda d: True)
    assert evaluated == ["sohbet-d"] and rep["sohbet"]["tag"] == "sohbet-d"
    assert not any(t == "sohbet-e" and "model.bin" in names for t, names in downloads)

    loc2 = {t: {"qa_score": 0.1, "persona_ok": True} for t in man}
    importer, _, _, evaluated = _fake_world(monkeypatch, tmp_path / "2", man, loc2, hold=False)
    rep = importer.run(None, lambda d: True)
    assert evaluated == ["sohbet-d", "sohbet-c", "sohbet-b"] and rep["changed"] is False
    assert "sınır" in rep["sohbet_adaylar"][-1]["neden"]


def test_local_eval_failure_rejects(monkeypatch, tmp_path):
    man = {"sohbet-a": {"val_ppl_int8": 12, "persona_ok": True, "qa_score": 0.9, "sha256": "a"}}
    importer, _, _, _ = _fake_world(monkeypatch, tmp_path, man, {}, hold=False)

    def boom(d):
        raise ImportError("evaluate_release yok")

    monkeypatch.setattr(importer, "_local_eval", boom)
    rep = importer.run(None, lambda d: True)
    assert rep["changed"] is False and "yerel değerlendirme yapılamadı" in rep["sohbet"]["neden"]
