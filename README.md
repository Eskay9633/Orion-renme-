# Orion — gelişim deposu

Bu depo **Orion'un beyninin eğitildiği** yerdir. Uygulamanın kendisi (işlem deposu) gizli depodadır;
oradan en yeni modeller buradaki **Releases** sayfasından alınır, güvenlik tabanından geçirilir ve
telefona giden APK'ya konur.

Neden ayrı? Halka açık depolarda GitHub Actions dakikaları ücretsiz ve sınırsızdır; Orion'un
sıfırdan eğitilen modelleri burada günlerce, kaldığı yerden devam ederek eğitilebilir.

## İki beyin

| | Ne yapar | İş akışı | Release etiketi |
|---|---|---|---|
| **Komut beyni** | Türkçe isteği makine diline çevirir (`{"cmd","args"}`) | `komut.yml` | `komut-*` |
| **Sohbet beyni** | Türkçe sohbet eden dil modeli (~17M parametre, sıfırdan) | `sohbet.yml` (6 saatte bir) | `sohbet-*` |

Hepsi Orion'a ait: kendi tokenizer'ı (bayt düzeyinde BPE, Türkçe ekleri ayıran ön bölme), kendi
küçük GPT'si, kendi veri üreticisi (`grammar.py`). Hazır model ağırlığı kullanılmaz.

## Veri ve lisans
- Komut beyni: yalnızca programla üretilen veri (lisans sorunu yok).
- Sohbet beyni: Türkçe Vikipedi (`wikimedia/wikipedia`, 20231101.tr — CC BY-SA 3.0 / GFDL; atıf:
  Vikipedi katkıcıları) + Orion'un kendi yazdığı kişilik diyalogları. Vikipedi metninin "aynı lisansla
  paylaş" şartının eğitilmiş modele geçip geçmediği hukuken net değildir — **doğrulanmadı**.
- Kullanıcının telefon verisi bu depoya **asla** gelmez; o veri yalnızca gizli işlem deposunda kalır.
- Bu depoda parola/anahtar yoktur.

## Kod nereden geliyor?
Kodun asıl yeri işlem deposudur; `tools/gelisim-sync.sh` ile buraya eşitlenir. Buradaki kodu
doğrudan değiştirme, işlem deposunda değiştir ve eşitle.

## Yerelde test
```bash
cd core-model && pip install torch numpy regex pytest pyarrow && python -m pytest -q
```
