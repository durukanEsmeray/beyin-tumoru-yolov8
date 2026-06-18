# YOLOv8 ile Beyin Tümörü Tespiti 🧠
 
  İstanbul Topkapı Üniversitesi — Derin Öğrenme Dersi Final Ödevi
  Bu projede, manyetik rezonans (MR) görüntülerinde beyin tümörünün otomatik tespiti için
  **YOLOv8n** nesne tespiti modeli, **transfer learning** yöntemiyle eğitilmiştir.
 
  ## 📋 Proje Özeti
 
  | | |
  |---|---|
  | **Görev** | Nesne Tespiti (Object Detection) |
  | **Model** | YOLOv8n (COCO ön-eğitimli, transfer learning) |
  | **Veri Seti** | [Ultralytics Brain Tumor](https://docs.ultralytics.com/tr/datasets/detect/brain-tumor/) (893 eğitim + 223     
  doğrulama) |
  | **Sınıflar** | `negative`, `positive` |
  | **Ortam** | Google Colab — Tesla T4 GPU |
 
  ## 📊 Sonuçlar (Validasyon Seti)
 
  | Metrik | Genel | negative | positive |
  |--------|:---:|:---:|:---:|
  | mAP@0.5 | 0.489 | 0.576 | 0.402 |
  | mAP@0.5:0.95 | 0.360 | 0.430 | 0.293 |
  | Precision | 0.459 | 0.549 | 0.369 |
  | Recall | 0.775 | 0.721 | 0.833 |
 
  **Model karşılaştırması (ablation):** YOLOv8n (0.489 mAP50) ile YOLOv8s (0.493 mAP50)
  neredeyse eşit sonuç vermiştir. Bu, performansı sınırlayan faktörün model kapasitesi değil,
  veri setinin küçüklüğü ve görevin doğal zorluğu olduğunu göstermektedir. Bu nedenle daha
  hızlı ve hafif olan **YOLOv8n** nihai model olarak seçilmiştir.
 
  ## 📁 Depo İçeriği
 
  ```
  beyin_tumoru_yolov8_final.ipynb   # Tüm adımları içeren çalıştırılabilir kod
  derin_ogrenme_final_rapor.pdf     # Detaylı proje raporu
  cikti/                            # Eğitim/değerlendirme çıktıları
    ├── results.png                 # Eğitim eğrileri
    ├── confusion_matrix.png        # Karışıklık matrisi
    ├── BoxPR_curve.png             # Precision-Recall eğrisi
    ├── ornek_tahminler.png         # Örnek tümör tespitleri
    └── weights/best.pt             # Eğitilmiş model ağırlıkları
  ```
 
  ## 🚀 Kullanım
 
  ```python
  !pip install ultralytics
  from ultralytics import YOLO
 
  # Veri setini indir ve eğit
  model = YOLO("yolov8n.pt")
  model.train(data="brain-tumor.yaml", epochs=50, imgsz=640, batch=16, seed=42)
 
  # Eğitilmiş modelle tahmin
  model = YOLO("cikti/weights/best.pt")
  results = model.predict("ornek_mr.jpg", conf=0.25)
  ```
 
  ## 🛠️ Kullanılan Teknolojiler
 
  Python 3.12 · Ultralytics YOLOv8 (8.4.69) · PyTorch 2.11 (CUDA 12.8) · OpenCV · Matplotlib
