# Basit Transformer Dil Modeli

Bu proje, sıfırdan bir Transformer tabanlı dil modeli oluşturmayı ve eğitmeyi amaçlamaktadır. Model, Türkçe metinler üzerinde eğitim alır ve metin üretimi yapabilir.

## **İçindekiler**

- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Örnek Çıktı](#örnek-çıktı)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## **Kurulum**

  1. Depoyu Klonlayın:
     ```bash
     git clone https://github.com/kullanıcı_adı/transformer-dil-modeli.git
     cd transformer-dil-modeli
     ```
  2. Gerekli Paketleri Kurun:
     Proje, numpy ve tqdm paketlerini kullanmaktadır. Aşağıdaki komutlarla paketleri kurabilirsiniz:
     ```bash
     pip install numpy tqdm
     ```
  4. Veri Dosyaları:
     Data klasörü içerisinde, modelin eğitimi için kullanacağınız Türkçe metin dosyalarını (.txt formatında) ekleyin.

## **Kullanım**

  1.Modeli eğitmek ve eğitilen modeli kaydetmek için:
      ```bash
      python main.py
      ```
  2.Eğitilen Modeli Yükleme ve Metin Üretme:
      ```bash
      python generate_text.py
      ```
## **Proje Yapısı**
  ```stylus
  .
  ├── data/
  │   ├── metin1.txt
  │   ├── metin2.txt
  │   └── ...
  ├── src/
  │   ├── tokenizer.py
  │   ├── model.py
  │   ├── utils.py
  │   ├── inference.py
  │   └── ...
  ├── main.py
  ├── generate_text.py
  └── README.md
  ```
  data/: Eğitim verisinin bulunduğu klasör.
  src/: Yardımcı modüllerin ve modelin bulunduğu klasör.
  tokenizer.py: Basit bir tokenizer sınıfı.
  model.py: Transformer modelinin tanımı.
  utils.py: Veri yükleme ve ön işleme fonksiyonları.
  inference.py: Metin üretimi için fonksiyonlar.
  main.py: Modelin eğitimi ve kaydedilmesi.
  generate_text.py: Eğitilen modelin yüklenmesi ve metin üretimi.
  README.md: Projenin açıklaması ve kullanım talimatları.

## **Örnek Çıktı**
  Seed Text: "Merhaba, nasılsın?"
  Generated Text: "Merhaba, nasılsın? bugün hava gerçekten güzel. arkadaşlarla parkta buluşup sohbet edeceğiz..."
## **Katkıda Bulunma**
  Katkıda bulunmak isterseniz, lütfen bir pull request oluşturun veya bir issue açın.
## **Lisans**
  MIT 2024
