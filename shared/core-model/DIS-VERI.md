# Dış veri kaynakları ve atıf (komut beyni)

Orion'un komut beyni (Türkçe → makine dili) çoğunlukla programla üretilmiş veriyle (`core-model/orion_core/datagen.py`)
eğitilir. Ek olarak aşağıdaki açık lisanslı veri kümeleri kullanılır. Dosyalar eğitim işinde sabit adres + sha256 ile
indirilir (`python -m orion_core.massive prepare --download`); depoya veri girmez.

## Amazon MASSIVE 1.1 — tr-TR

- Kaynak: https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
- sha256: `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`
- Lisans: **CC BY 4.0** (Creative Commons Attribution 4.0 International) — doğrulandı: tarball içindeki `1.1/LICENSE`
  ve github.com/alexa/massive `NOTICE.md`. Lisans metni: https://creativecommons.org/licenses/by/4.0/
- Atıf: MASSIVE © Amazon.com, Inc. — Jack FitzGerald ve ark. (2022), "MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset with 51 Typologically-Diverse Languages". SLURP (Bastianelli ve ark. 2020, CC BY 4.0)
  üzerine kuruludur.
- Değişiklikler: yalnızca tr-TR satırları; İngilizce uyandırma sözcüğü ("olly", "alexa") silindi; değerlendirici
  yargılarına göre kalite süzgeci; niyet + yuvalar Orion makine diline çevrildi (çevrilemeyen satırlar atıldı);
  Orion'da komutu olmayan niyetler "kapsam dışı" (`ask`) örneğine dönüştürüldü.
- Kullanım: `train` bölümü eğitimde (akışın %12'si); `dev` + `test` bölümleri yalnızca **rapor** amaçlı
  değerlendirme kümesi `massive_tr` (kapı eşiklerine girmez).

## xSID 0.6 — tr

- Kaynak: https://github.com/mainlp/xsid (commit `761d926c3f2e9f6bea13192d2f39cec1d04d9777`),
  `data/xSID-0.6/tr.valid.conll` (sha256 `0ac8d0d818c65eefd332f17d01f51e272ab1fce630216b7aa529291e716b83fc`),
  `data/xSID-0.6/tr.test.conll` (sha256 `c600eb35a716f28d682fe77610dc30a8f9e63612df48ebb52ed3f9e5e0f2cf21`)
- Lisans: **CC BY-SA 4.0** — doğrulandı: depodaki `LICENSE`. Lisans metni: https://creativecommons.org/licenses/by-sa/4.0/
- Atıf: Rob van der Goot ve ark. (2021), "From Masked Language Modeling to Translation: Non-English Auxiliary Tasks
  Improve Zero-shot Spoken Language Understanding" (NAACL 2021).
- Kullanım: **yalnızca değerlendirme** (`xsid_tr`, yalnızca rapor); eğitime girmez. Bu veriden türetilen
  değerlendirme dosyası da CC BY-SA 4.0 lisanslıdır.

## Sızıntı kuralı

MASSIVE `dev`/`test`, xSID ve depodaki elle yazılmış/kör ölçüm kümelerinde (normalize edilmiş hâliyle) geçen hiçbir cümle
eğitime girmez; eğitim yükleyicisi bunu her çalışmada yeniden denetler, çakışma varsa iş durur
(`core-model/tests/test_massive.py`).

# Dış veri kaynakları ve atıf (sohbet beyni)

## MKQA — tr

- Kaynak: https://raw.githubusercontent.com/apple/ml-mkqa/f964fe1bac4d580cee3d4928572df1f676f8e656/dataset/mkqa.jsonl.gz
  (github.com/apple/ml-mkqa, commit `f964fe1bac4d580cee3d4928572df1f676f8e656`)
- sha256: `f77ac6bf6306e98816c5c047880db71a3c67d2f1491b169896d63e9bb14ef3d3`
- Lisans: **CC BY-SA 3.0** (veri; kod Apache-2.0) — doğrulandı: depo README "License" bölümü ("dataset is licensed
  under CC BY-SA 3.0 Unported"). Lisans metni: https://creativecommons.org/licenses/by-sa/3.0/
- Atıf: MKQA © Apple Inc. — Shayne Longpre, Yi Lu, Joachim Daiber (2021), "MKQA: A Linguistically Diverse Benchmark for
  Multilingual Open Domain Question Answering" (TACL). Natural Questions (Google, CC BY-SA 3.0) sorularının çevirisidir.
- Değişiklikler: yalnızca `tr` sorgu ve cevapları; `long_answer` ve boş/bozuk cevaplar atıldı; kısa cevap Orion'un
  sesiyle bir cümleye sarıldı ("Bildiğim kadarıyla: X.") — cevaba veride olmayan bilgi EKLENMEZ; sayılar Türkçe ondalıkla,
  tarihler Türkçe ay adıyla yazıldı; zamana bağlı sorulara "bu bilgi değişmiş olabilir" notu eklendi; en çok 400
  `unanswerable` soru dürüst "bilmiyorum, araştırayım" cevabına çevrildi; sınav soruları ve yakın biçimleri atıldı.
- Kullanım: ~6.100 soru sohbet beyninin diyalog akışında (eğitim, `chat.prepare`; indirme sabit adres + sha256, uymazsa
  iş durur); 1.000 soru ayrılır (`mkqa-eval.jsonl`) ve yalnızca **rapor** amaçlı değerlendirilir (manifest
  `mkqa_eval`, kapı eşiği yok). Türetilen eğitim verisi de CC BY-SA 3.0 lisanslıdır.
