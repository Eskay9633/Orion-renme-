# Çeviri verisi denetim klasörü

`hedef.txt` dosyasını bu klasöre "ÇİFT PARÇA" biçiminde (ör. `100000 3`) yazıp main'e işlemek
`ceviri` iş akışını (.github/workflows/cevir.yml) başlatır. Dosya değişmezse yeniden başlamaz;
yeniden çalıştırmak için sayıyı değiştir ya da Actions → ceviri → Run workflow kullan.

Çıktı: `ceviri-YYYYMMDD-HHMM` ön sürüm Release'i (`ceviri.jsonl.gz`, `stats.json`, `LISANS-ATIF.md`).
Kaggle (kaggle.yml) en yeni `ceviri-*` Release'ini kendiliğinden kullanır (hedef.txt'te üçüncü alan `yok` ile kapatılır).
Veri depoya işlenmez.
