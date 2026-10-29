"""İçe aktarma kararları (ağ yok)."""
from orion_core.importer import chat_ok, komut_better

FLOOR = {"val_ppl_max": 60, "persona_ok": True}


def test_komut_regression_rejected():
    # Elle kuralı yalnızca aynı kümede ölçülmüş sayılar arasında (bkz. test_handwritten_rule_*)
    s = "handwritten_set_sha256"
    cur = {"sha256": "a", "eval": {"handwritten_exact": 0.96, "t2c_ood_exact": 0.67, s: "S"}}
    assert komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.97, "t2c_ood_exact": 0.70, s: "S"}}, cur)[0]
    assert not komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.93, "t2c_ood_exact": 0.80, s: "S"}}, cur)[0]
    assert not komut_better({"sha256": "b", "eval": {"handwritten_exact": 0.93, "t2c_ood_exact": 0.80}}, cur,
                            local_hw=(0.93, 0.96))[0]
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


# ---- Elle yazılmış küme değişince adil karşılaştırma (Tur 18) ----

def _m(sha, hw, ood=0.7, set_sha=None):
    ev = {"handwritten_exact": hw, "t2c_ood_exact": ood}
    if set_sha:
        ev["handwritten_set_sha256"] = set_sha
    return {"sha256": sha, "eval": ev}


def test_handwritten_rule_strict_on_same_set():
    """Aynı kümede toleranssız: bir cümle bile geri giderse alınmaz."""
    cur = _m("a", 0.9655, set_sha="S1")
    assert not komut_better(_m("b", 0.9310, set_sha="S1"), cur)[0]
    assert komut_better(_m("b", 0.9655, set_sha="S1"), cur)[0]
    # yerel ölçüm varsa o tercih edilir (iki model de bu depoda aynı kümede ölçüldü)
    ok, why = komut_better(_m("b", 0.9310, set_sha="S1"), cur, local_hw=(0.99, 0.5))
    assert ok and "yeniden ölçüldü" in why
    assert not komut_better(_m("b", 0.99, set_sha="S1"), cur, local_hw=(0.90, 0.95))[0]
    # yerel ölçüm yapılamadıysa aynı kümede manifest sayıları katı karşılaştırılır
    assert not komut_better(_m("b", 0.9310, set_sha="S1"), cur, local_hw_error="torch yok")[0]
    ok, why = komut_better(_m("b", 0.9655, set_sha="S1"), cur, local_hw_error="torch yok")
    assert ok and "manifest" in why


def test_handwritten_rule_fails_closed_across_different_sets_without_local_measurement():
    """Farklı kümelerin sayıları karşılaştırılmaz; yerel ölçüm de yoksa model ALINMAZ (kapalı kalır)."""
    old = _m("a", 0.9655)  # eski manifest: küme kimliği yok
    ok, why = komut_better(_m("b", 0.99, set_sha="S2"), old)
    assert not ok and why.startswith("elle yazılmış küme ölçülemedi: ")
    ok, why = komut_better(_m("b", 0.99, set_sha="S2"), _m("a", 0.9655, set_sha="S1"), local_hw_error="torch yok")
    assert not ok and why == "elle yazılmış küme ölçülemedi: torch yok"
    # yerel ölçümle karşılaştırılır; yeni cümle yapıları kuralı yine geçerli
    assert komut_better(_m("b", 0.90, set_sha="S2"), old, local_hw=(0.9, 0.9))[0]
    assert not komut_better(_m("b", 0.99, ood=0.60, set_sha="S2"), _m("a", 0.9, ood=0.70, set_sha="S1"),
                            local_hw=(0.95, 0.9))[0]


def test_handwritten_rule_uses_local_same_set_measurement():
    old, new = _m("a", 0.9655), _m("b", 0.80, set_sha="S2")
    assert komut_better(new, old, local_hw=(0.92, 0.88))[0]
    ok, why = komut_better(new, old, local_hw=(0.87, 0.88))
    assert not ok and "aynı kümede" in why


def _komut_world(monkeypatch, tmp_path, new_m: dict, cur_m: dict):
    import json
    from orion_core import importer
    shared = tmp_path / "shared"
    core = shared / "core-model"
    core.mkdir(parents=True)
    (shared / "chat-model-floor.json").write_text(json.dumps({"val_ppl_max": 60, "persona_ok": True}), encoding="utf-8")
    (core / "manifest.json").write_text(json.dumps(cur_m), encoding="utf-8")
    (core / "model.bin").write_text("eski", encoding="utf-8")
    monkeypatch.setattr(importer, "SHARED", shared)
    monkeypatch.setattr(importer, "latest", lambda prefix, token: {"tag_name": "komut-x", "assets": []})
    monkeypatch.setattr(importer, "recent", lambda prefix, token, n=8: [])

    def fake_download(rel, names, dest, token):
        for n in names:
            (dest / n).write_text(json.dumps(new_m) if n == "manifest.json" else f"yeni:{n}", encoding="utf-8")

    monkeypatch.setattr(importer, "download", fake_download)
    return importer, core


def test_run_measures_both_models_locally_when_sets_differ(monkeypatch, tmp_path):
    importer, core = _komut_world(monkeypatch, tmp_path, _m("b", 0.80, set_sha="S2"), _m("a", 0.9655))
    measured = []

    def measure(d):
        body = (d / "model.bin").read_text()
        measured.append(body)
        return 0.93 if body.startswith("yeni") else 0.90

    rep = importer.run(None, lambda d: True, measure_handwritten=measure)
    assert sorted(measured) == ["eski", "yeni:model.bin"]
    assert rep["komut"]["alındı"] and rep["changed"]
    assert rep["elle_yerel"] == {"yeni": 0.93, "mevcut": 0.9}
    assert (core / "model.bin").read_text() == "yeni:model.bin"


def test_run_rejects_local_regression_and_respects_floor(monkeypatch, tmp_path):
    importer, core = _komut_world(monkeypatch, tmp_path, _m("b", 0.99, set_sha="S2"), _m("a", 0.9655))
    rep = importer.run(None, lambda d: True,
                       measure_handwritten=lambda d: 0.85 if (d / "model.bin").read_text().startswith("yeni") else 0.90)
    assert not rep["komut"]["alındı"] and (core / "model.bin").read_text() == "eski"
    # Elle kümede iyi olsa da güvenlik tabanını geçemeyen model alınmaz.
    importer, core = _komut_world(monkeypatch, tmp_path / "2", _m("b", 0.99, set_sha="S2"), _m("a", 0.9655))
    rep = importer.run(None, lambda d: False, measure_handwritten=lambda d: 0.95)
    assert not rep["komut"]["alındı"] and "güvenlik tabanını" in rep["komut"]["neden"]
    assert (core / "model.bin").read_text() == "eski"


def test_run_fails_closed_when_local_measurement_fails(monkeypatch, tmp_path):
    """Kümeler farklı ve yerel ölçüm yapılamadı: güvenlik tabanı geçse bile model ALINMAZ."""
    importer, core = _komut_world(monkeypatch, tmp_path, _m("b", 0.99, set_sha="S2"), _m("a", 0.9655))

    def boom(d):
        raise RuntimeError("torch yok")

    rep = importer.run(None, lambda d: True, measure_handwritten=boom)
    assert not rep["komut"]["alındı"] and not rep["changed"]
    assert rep["komut"]["neden"] == "elle yazılmış küme ölçülemedi: torch yok"
    assert rep["elle_yerel"] == {"hata": "torch yok"} and (core / "model.bin").read_text() == "eski"


def test_run_same_set_remeasures_locally_and_prefers_it(monkeypatch, tmp_path):
    importer, core = _komut_world(monkeypatch, tmp_path, _m("b", 0.99, set_sha="S1"), _m("a", 0.96, set_sha="S1"))
    rep = importer.run(None, lambda d: True,
                       measure_handwritten=lambda d: 0.85 if (d / "model.bin").read_text().startswith("yeni") else 0.90)
    assert not rep["komut"]["alındı"] and "aynı kümede" in rep["komut"]["neden"]
    assert rep["elle_yerel"] == {"yeni": 0.85, "mevcut": 0.9} and (core / "model.bin").read_text() == "eski"


def test_run_same_set_falls_back_to_strict_manifest(monkeypatch, tmp_path):
    importer, _ = _komut_world(monkeypatch, tmp_path, _m("b", 0.95, set_sha="S1"), _m("a", 0.96, set_sha="S1"))

    def boom(d):
        raise RuntimeError("torch yok")

    rep = importer.run(None, lambda d: True, measure_handwritten=boom)
    assert not rep["komut"]["alındı"] and "geriledi" in rep["komut"]["neden"]
    importer, core = _komut_world(monkeypatch, tmp_path / "2", _m("b", 0.97, set_sha="S1"), _m("a", 0.96, set_sha="S1"))
    rep = importer.run(None, lambda d: True, measure_handwritten=boom)
    assert rep["komut"]["alındı"] and "torch yok" in rep["komut"]["neden"]
    assert (core / "model.bin").read_text() == "yeni:model.bin"


def test_manifest_records_handwritten_set_identity():
    """Manifest, puanın hangi kümede ölçüldüğünü yazar; kimlik yalnızca cümle + beklenen çıktıya bağlıdır."""
    import inspect
    from orion_core import manifest
    from orion_core.evaluate import handwritten_set_sha, load_handwritten
    src = inspect.getsource(manifest.write_manifest)
    assert "handwritten_set_sha256" in src and "handwritten_n" in src
    rows = load_handwritten()
    meta_changed = [dict(r, date="2000-01-01") for r in rows]
    assert handwritten_set_sha(rows) == handwritten_set_sha(meta_changed)
    assert handwritten_set_sha(rows) != handwritten_set_sha(rows[:29])
