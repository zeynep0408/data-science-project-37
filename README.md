# Data Science Project 37 — El Yazısı Rakam Tanıma (SVM + GridSearchCV) — CAPSTONE

**Modül**: ML-04 (Sınıflandırma 2) • **Tip**: CAPSTONE • **Süre**: 4-5 saat

## 🎯 Proje Senaryosu

Bir posta/form okuma sistemi için **data scientist** olarak çalışıyorsun. Sistem, üzerinde el yazısı rakamlar (0-9) bulunan zarfları ve formları tarıyor — posta kodları, fatura numaraları, anket kutucukları. Senden bu küçük rakam görüntülerini otomatik tanıyan bir model isteniyor.

Her rakam **8×8 piksellik küçük bir gri ton görüntü** olarak geliyor. Yani her örnek 64 sayıya (piksel) açılmış durumda. Bu **yüksek boyutlu** (64 feature) bir problem — ve burası tam olarak **SVM'in (Support Vector Machine) parladığı** yer.

Bu, ML-04'ün **CAPSTONE** projesi. Modülde öğrendiğin HER ŞEYİ tek bir akışta birleştiriyorsun:
- ✅ **SVM** — kernel seçimi (linear / rbf / poly), C ve gamma'nın etkisi
- ✅ **StandardScaler** — SVM mesafe/iç-çarpım tabanlı → scaling ZORUNLU
- ✅ **Pipeline** ile data leakage'i önleme
- ✅ **GridSearchCV** — C × gamma hiperparametre taraması
- ✅ **Cross-validation** — StratifiedKFold ile sağlam değerlendirme
- ✅ **Model Selection** — 5 algoritmayı yarıştırıp en iyisini seçme (NB / DT / RF / SVM / LR)
- ✅ **Multi-class** — 10 sınıf (one-vs-one), `f1_macro`, 10×10 confusion matrix

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-37

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

> ⚠️ **macOS notu**: GridSearchCV ve diğer modellerde `n_jobs=-1` **KULLANMA**. macOS'ta paralel worker (loky) çökebiliyor. Kodda `n_jobs=1` yazılı — değiştirme.

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 717 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📊 Veri Seti

**Kaynak**: `sklearn.datasets.load_digits` (built-in). Orijinali: UCI Machine Learning Repository — *Optical Recognition of Handwritten Digits* (E. Alpaydin & C. Kaynak). **İnternet gerekmez** — veri sklearn içinde gömülü gelir.

```python
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
```

**Boyut**: 1797 örnek × 64 feature
**Görüntü formatı**: her örnek **8×8 = 64 piksel**, gri ton değerleri **0-16** arası (0 = beyaz/boş, 16 = en koyu)
**Hedef**: 10 sınıf — rakamlar **0-9** (**çok sınıflı / multi-class** problem)
**Sınıf dağılımı**: her rakamdan ~180 örnek (yaklaşık dengeli)

### Bir Örneğin Matris Gösterimi

`X[0]` (etiketi `0` olan bir rakam) 64 sayıdan oluşur; `.reshape(8, 8)` ile görüntüye geri açılır:

```
 0  0  5 13  9  1  0  0
 0  0 13 15 10 15  5  0
 0  3 15  2  0 11  8  0
 0  4 12  0  0  8  8  0
 0  5  8  0  0  9  8  0
 0  4 11  0  1 12  7  0
 0  2 14  5 10 12  0  0
 0  0  6 13 10  0  0  0
```

Yüksek değerler (koyu pikseller) rakamın çizgilerini oluşturur — yukarıdaki matriste bir **0** rakamının halka şekli seçilebiliyor. `visualize_sample(X, y, index)` bunu `plt.imshow` ile çizer.

### Feature Yapısı

| Özellik | Açıklama |
|---------|----------|
| Feature sayısı | 64 (8×8 görüntünün düzleştirilmiş / flatten hali) |
| Her feature | bir pikselin gri ton yoğunluğu (0-16) |
| Ölçek | tüm pikseller aynı birim → yine de SVM için **StandardScaler şart** |

### Domain Notu

Bu veri seti, optik karakter tanıma (OCR) araştırmalarının klasik bir benchmark'ıdır. Gerçek dünyada (posta kodu okuma, banka çeki işleme) görüntüler çok daha yüksek çözünürlüklü olur; burada 8×8'e küçültülmüş hali kullanılır — yine de 64 boyut, SVM gibi yüksek boyutta güçlü algoritmaları sergilemek için yeterince zorlayıcıdır. **Multi-class (10 sınıf)** olduğu için SVM içsel olarak **one-vs-one** strateji kullanır.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **15 fonksiyonu** sırayla doldur. Her fonksiyon, ilgili testler pass olana kadar düzenlenmeli.

1. **`load_digits_data()`** — `load_digits(return_X_y=True)` ile X, y yükle
2. **`explore_data(X, y)`** — örnek/feature/sınıf sayısı, sınıf dağılımı
3. **`visualize_sample(X, y, index)`** — bir örneği 8×8 görüntüye çevirip çiz
4. **`split_data(X, y)`** — train/test böl (stratify, random_state=42)
5. **`build_svm_pipeline(kernel, C, gamma)`** — `StandardScaler` + `SVC` Pipeline
6. **`train_model(pipe, X_train, y_train)`** — fit et
7. **`evaluate_model(pipe, X_test, y_test)`** — accuracy, f1_macro, confusion matrix
8. **`compare_kernels(...)`** — linear / rbf / poly karşılaştır
9. **`grid_search_svm(X_train, y_train)`** — GridSearchCV (C × gamma, `n_jobs=1`)
10. **`get_best_params(grid)`** — `grid.best_params_`
11. **`cross_validate_model(pipe, X, y, cv)`** — StratifiedKFold CV (mean, std)
12. **`compare_five_algorithms(...)`** — NB / DT / RF / SVM / LR yarıştır
13. **`predict_digit(pipe, image_row)`** — tek görüntü için tahmin
14. **`evaluate_at_best(...)`** — GridSearch best model'i test'te değerlendir
15. **`run_pipeline()`** — tüm akışı uçtan uca çalıştır

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] `load_digits` ile gömülü veri setini yükleyebileceksin
- [x] Görüntü → flatten (64 feature) mantığını anlayacaksın
- [x] SVM'in yüksek boyutta neden güçlü olduğunu göreceksin
- [x] **SVC kernel** (rbf / linear / poly) ve **C / gamma** etkisini kavrayacaksın
- [x] **GridSearchCV** ile hiperparametre taraması yapacaksın
- [x] **cross_val_score** + **StratifiedKFold** ile sağlam değerlendirme öğreneceksin
- [x] `best_params_` / `best_score_` / `best_estimator_` kullanımını pratik yapacaksın
- [x] **5-algoritma karşılaştırma** pipeline'ı ile model seçimi yapacaksın
- [x] **Multi-class** metriklerini (f1 macro, one-vs-one) yorumlayacaksın
- [x] SVM gibi mesafe/iç-çarpım tabanlı modellerde **scaling**'in zorunlu olduğunu göreceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (16+ test)

Tümü pass olmalı:
- Veri yükleme + shape (1797, 64), piksel aralığı 0-16, 10 sınıf
- `explore_data` doğru özet (n_classes=10, image_shape=(8,8))
- `split_data` boyutları (train 1437 / test 360) + stratify
- Pipeline doğru tip + SVC parametreleri
- `evaluate_model` → 10×10 confusion matrix, accuracy > 0.95
- `compare_kernels` → 3 kernel hepsi > 0.9
- `grid_search_svm` → **n_jobs == 1**, best_score > 0.95
- `cross_validate_model` → mean > 0.95
- `compare_five_algorithms` → 5 model, en iyisi > 0.95
- `predict_digit` → bilinen örnekte doğru tahmin
- `run_pipeline` → grid_test_accuracy > 0.97, n_classes == 10

## 📊 Beklenen Sonuçlar

```
Örnek: 1797, Feature: 64 (8×8), Sınıf: 10 (0-9)
Train: 1437, Test: 360 (stratify ile her sınıftan ~36)
Kernel karşılaştırması: linear ~0.97, rbf ~0.98, poly ~0.98
GridSearch best: genellikle C=10, gamma='scale' → CV ~0.98
GridSearch test accuracy: ~0.98-0.99
5-algoritma kıyası: SVM en iyi (~0.98), ardından LR/RF (~0.97), NB en düşük
```

## 💡 İpuçları

- SVM iç-çarpım/mesafe tabanlı → **scaling unutursan performans düşer** (Pipeline kullan)
- Pipeline adımı `'svm'` → GridSearch parametre öneki `'svm__C'`, `'svm__gamma'`
- `gamma='scale'` iyi bir varsayılan; çok büyük gamma overfit eder
- `cross_val_score(..., cv=StratifiedKFold(...))` → dengeli fold'lar
- Multi-class → `f1_score(..., average='macro')`
- `grid.best_estimator_` → en iyi parametrelerle yeniden eğitilmiş model
- `random_state=42` her yerde — tekrarlanabilirlik

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olabilir)
- **`n_jobs=-1` KULLANMA** — `n_jobs=1` kalsın (macOS loky çökmesi)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
