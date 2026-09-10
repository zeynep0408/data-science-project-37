"""
DS-37 — El Yazısı Rakam Tanıma (SVM + GridSearchCV) — CAPSTONE

Bir posta/form okuma sistemi için el yazısı rakamları (0-9) tanıyan bir
model kuracaksın. Veri 8×8 piksellik küçük gri ton görüntülerden oluşuyor
(her görüntü 64 sayıya açılmış). 64 feature → yüksek boyut → SVM'in
parladığı yer. SVM mesafe/iç-çarpım tabanlı olduğu için scaling ZORUNLU.

Bu CAPSTONE'da ML-04'ün HEPSİNİ kullanıyorsun:
- SVM kernel seçimi (linear / rbf / poly)
- C ve gamma tuning → GridSearchCV
- Cross-validation (StratifiedKFold)
- 5 algoritmayı (NB / DT / RF / SVM / LR) yarıştırıp en iyisini seçme

⚠️ ÖNEMLİ: GridSearchCV ve hiçbir yerde n_jobs=-1 KULLANMA. n_jobs=1 yaz
(veya hiç yazma). macOS'ta paralel worker (loky) çökebiliyor.

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Veri setini yükle
def load_digits_data():
    X, y = load_digits(return_X_y=True)
    return X, y
pass


# 2. Veriyi keşfet
def explore_data(X, y):
    classes, counts = np.unique(y, return_counts=True)
    return {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(classes),
        'image_shape': (8, 8),
        'class_distribution': {int(c): int(n) for c, n in zip(classes, counts)},
    }
    pass


# 3. Örnek bir rakamı görselleştir
def visualize_sample(X, y, index=0):
    image = X[index].reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.title(f"Etiket: {y[index]}")
    plt.axis('off')
    return None
    pass


# 4. Train / test böl
def split_data(X, y):
   
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pass


# 5. SVM Pipeline kur
def build_svm_pipeline(kernel='rbf', C=1, gamma='scale'):
      return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma)),
    ])
      pass


# 6. Modeli eğit
def train_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe
    pass


# 7. Modeli değerlendir
def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    pass


# 8. Kernel'leri karşılaştır
def compare_kernels(X_train, X_test, y_train, y_test):
    results = {}
    for kernel in ['linear', 'rbf', 'poly']:
        pipe = build_svm_pipeline(kernel=kernel)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results[kernel] = accuracy_score(y_test, y_pred)
    return results
    pass


# 9. GridSearchCV ile C × gamma tuning
def grid_search_svm(X_train, y_train):
    pipe = build_svm_pipeline()
    param_grid = {
        'svm__C': [1, 10],
        'svm__gamma': ['scale', 0.01, 0.001],
    }
    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring='accuracy', n_jobs=1,
    )
    grid.fit(X_train, y_train)
    return grid

    pass


# 10. En iyi parametreleri al
def get_best_params(grid):
    return grid.best_params_
    pass


# 11. Cross-validation
def cross_validate_model(pipe, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
    }
    pass


# 12. 5 algoritmayı karşılaştır
def compare_five_algorithms(X_train, X_test, y_train, y_test):
    models = {
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale'),
        'LR': LogisticRegression(max_iter=5000),
    }
    results = []
    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results.append({
            'model': name,
            'test_accuracy': accuracy_score(y_test, y_pred),
        })
    return results  
    pass


# 13. Tek görüntü için tahmin
def predict_digit(pipe, image_row):
    X_new = np.array(image_row).reshape(1, -1)
    pred = int(pipe.predict(X_new)[0])
    return {'predicted_digit': pred}

    pass


# 14. GridSearch best model'i test'te değerlendir
def evaluate_at_best(X_train, X_test, y_train, y_test):
    grid = grid_search_svm(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    return {
        'best_params': grid.best_params_,
        'test_accuracy': accuracy_score(y_test, y_pred),
    }
    pass


# 15. Tüm pipeline'ı uçtan uca çalıştır
def run_pipeline():
    X, y = load_digits_data()
    info = explore_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Kernel karşılaştırması
    kernel_scores = compare_kernels(X_train, X_test, y_train, y_test)
    best_kernel = max(kernel_scores, key=kernel_scores.get)

    # GridSearch
    grid = grid_search_svm(X_train, y_train)
    grid_best_score = float(grid.best_score_)
    best_model = grid.best_estimator_
    grid_test_accuracy = accuracy_score(y_test, best_model.predict(X_test))

    # 5-algoritma karşılaştırması
    five = compare_five_algorithms(X_train, X_test, y_train, y_test)
    best_algorithm = max(five, key=lambda d: d['test_accuracy'])['model']

    return {
        'n_samples': info['n_samples'],
        'n_classes': info['n_classes'],
        'best_kernel': best_kernel,
        'grid_best_score': grid_best_score,
        'grid_test_accuracy': grid_test_accuracy,
        'best_algorithm': best_algorithm,
    }
    pass


if __name__ == "__main__":
    result = run_pipeline()
    print("📊 Pipeline Sonuçları:")
    print(f"  Örnek sayısı       : {result['n_samples']}")
    print(f"  Sınıf sayısı       : {result['n_classes']}")
    print(f"  En iyi kernel      : {result['best_kernel']}")
    print(f"  Grid best CV score : {result['grid_best_score']:.4f}")
    print(f"  Grid test accuracy : {result['grid_test_accuracy']:.4f}")
    print(f"  En iyi algoritma   : {result['best_algorithm']}")
