import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_digits_data, explore_data, visualize_sample, split_data,
    build_svm_pipeline, train_model, evaluate_model, compare_kernels,
    grid_search_svm, get_best_params, cross_validate_model,
    compare_five_algorithms, predict_digit, evaluate_at_best, run_pipeline,
)


# Yardımcı — eğitilmiş pipeline oluştur (testler için)
def _build_trained_pipeline(kernel='rbf', C=10, gamma='scale'):
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = build_svm_pipeline(kernel=kernel, C=C, gamma=gamma)
    train_model(pipe, X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test


# 1. load_digits_data
def test_load_digits_data_shapes():
    X, y = load_digits_data()
    assert X.shape == (1797, 64)
    assert y.shape == (1797,)


def test_load_digits_data_value_range():
    X, y = load_digits_data()
    # Piksel değerleri 0-16 gri ton
    assert X.min() >= 0
    assert X.max() <= 16
    # 10 sınıf
    assert set(np.unique(y)) == set(range(10))


# 2. explore_data
def test_explore_data_structure():
    X, y = load_digits_data()
    info = explore_data(X, y)
    assert info['n_samples'] == 1797
    assert info['n_features'] == 64
    assert info['n_classes'] == 10
    assert info['image_shape'] == (8, 8)
    assert len(info['class_distribution']) == 10
    # Sınıf dağılımı toplamı tüm örnekleri kapsamalı
    assert sum(info['class_distribution'].values()) == 1797


# 3. visualize_sample
def test_visualize_sample_runs():
    import matplotlib
    matplotlib.use("Agg")
    X, y = load_digits_data()
    # Hata fırlatmadan çalışmalı, None dönmeli
    result = visualize_sample(X, y, index=0)
    assert result is None


# 4. split_data
def test_split_data_sizes():
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # 1797 * 0.2 = 359.4 → test 360, train 1437
    assert X_train.shape[0] == 1437
    assert X_test.shape[0] == 360
    assert X_train.shape[1] == 64


def test_split_data_stratified():
    X, y = load_digits_data()
    _, _, _, y_test = split_data(X, y)
    # Tüm 10 sınıf test setinde temsil edilmeli
    assert len(np.unique(y_test)) == 10


# 5. build_svm_pipeline
def test_build_svm_pipeline_steps():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    pipe = build_svm_pipeline(kernel='linear', C=5, gamma='scale')
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.named_steps['scaler'], StandardScaler)
    assert isinstance(pipe.named_steps['svm'], SVC)
    assert pipe.named_steps['svm'].kernel == 'linear'
    assert pipe.named_steps['svm'].C == 5


# 6-7. train_model + evaluate_model
def test_evaluate_model_returns_dict():
    pipe, _, X_test, _, y_test = _build_trained_pipeline()
    result = evaluate_model(pipe, X_test, y_test)
    for key in ['accuracy', 'f1_macro', 'confusion_matrix']:
        assert key in result
    # 10 sınıf → 10x10 confusion matrix
    assert result['confusion_matrix'].shape == (10, 10)
    # SVM + scaling digits'te çok güçlü
    assert result['accuracy'] > 0.95


# 8. compare_kernels
def test_compare_kernels():
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    scores = compare_kernels(X_train, X_test, y_train, y_test)
    assert set(scores.keys()) == {'linear', 'rbf', 'poly'}
    # Hepsi yüksek accuracy vermeli
    for k, acc in scores.items():
        assert acc > 0.9


# 9. grid_search_svm
def test_grid_search_svm():
    X, y = load_digits_data()
    X_train, _, y_train, _ = split_data(X, y)
    grid = grid_search_svm(X_train, y_train)
    # n_jobs=1 olmalı (macOS loky sorunu)
    assert grid.n_jobs == 1
    assert grid.best_score_ > 0.95
    assert 'svm__C' in grid.best_params_
    assert 'svm__gamma' in grid.best_params_


# 10. get_best_params
def test_get_best_params():
    X, y = load_digits_data()
    X_train, _, y_train, _ = split_data(X, y)
    grid = grid_search_svm(X_train, y_train)
    params = get_best_params(grid)
    assert isinstance(params, dict)
    assert params['svm__C'] in [1, 10]


# 11. cross_validate_model
def test_cross_validate_model():
    X, y = load_digits_data()
    pipe = build_svm_pipeline(kernel='rbf', C=10, gamma='scale')
    result = cross_validate_model(pipe, X, y, cv=5)
    assert 'mean' in result
    assert 'std' in result
    assert result['mean'] > 0.95


# 12. compare_five_algorithms
def test_compare_five_algorithms():
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = compare_five_algorithms(X_train, X_test, y_train, y_test)
    # list of dict olarak ele al
    results = list(results) if not isinstance(results, list) else results
    names = {r['model'] for r in results}
    assert names == {'NB', 'DT', 'RF', 'SVM', 'LR'}
    for r in results:
        assert 0 <= r['test_accuracy'] <= 1
    # En iyi algoritma yüksek accuracy vermeli
    best = max(results, key=lambda d: d['test_accuracy'])
    assert best['test_accuracy'] > 0.95


# 13. predict_digit
def test_predict_digit_structure():
    pipe, X_train, X_test, _, _ = _build_trained_pipeline()
    result = predict_digit(pipe, X_test[0])
    assert 'predicted_digit' in result
    assert isinstance(result['predicted_digit'], int)
    assert 0 <= result['predicted_digit'] <= 9


def test_predict_digit_correct_on_known():
    pipe, X_train, X_test, _, y_test = _build_trained_pipeline()
    # İyi eğitilmiş model bilinen örnekte doğru tahmin yapmalı
    result = predict_digit(pipe, X_test[0])
    assert result['predicted_digit'] == int(y_test[0])


# 14. evaluate_at_best
def test_evaluate_at_best():
    X, y = load_digits_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    result = evaluate_at_best(X_train, X_test, y_train, y_test)
    assert 'best_params' in result
    assert 'test_accuracy' in result
    assert result['test_accuracy'] > 0.97


# 15. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    for key in ['n_samples', 'n_classes', 'best_kernel',
                'grid_best_score', 'grid_test_accuracy', 'best_algorithm']:
        assert key in result
    assert result['n_samples'] == 1797
    assert result['n_classes'] == 10
    assert result['best_kernel'] in ['linear', 'rbf', 'poly']
    assert result['best_algorithm'] in ['NB', 'DT', 'RF', 'SVM', 'LR']
    # Digits + SVM + GridSearch çok iyi sonuç verir
    assert result['grid_test_accuracy'] > 0.97


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
