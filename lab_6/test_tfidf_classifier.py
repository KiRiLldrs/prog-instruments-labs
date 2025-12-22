import pytest
import pandas as pd
import tempfile
import os

from tfidf_classifier import TFIDFClassifier


@pytest.fixture
def train_df():
    return pd.DataFrame({
        "text": [
            "Продам квартиру в центре",
            "Ищу репетитора по математике",
            "Водитель со стажем",
            "Курсы Python для начинающих",
            "Ремонт компьютеров",
            "Продам гараж",
            "Репетитор по физике",
            "Курьер на полный день",
            "Продвинутый Python",
            "Основы C++",
            "Ремонт телефонов",
            "Аренда офиса"
        ],
        "subrubric": [
            "Недвижимость", "Образование", "Работа",
            "Образование", "Услуги",
            "Недвижимость", "Образование", "Работа",
            "Образование", "Образование",
            "Услуги", "Недвижимость"
        ]
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "text": ["Нужен Python-разработчик", "Сдам квартиру посуточно"],
        "subrubric": ["Работа", "Недвижимость"]
    })


def test_init_default_params():
    """
    Проверяет, что конструктор устанавливает параметры по умолчанию:
    - max_features=10000,
    - ngram_range=(1, 2),
    - min_class_size=50,
    - is_trained=False.
    """
    clf = TFIDFClassifier()
    assert clf.vectorizer.max_features == 10000
    assert clf.vectorizer.ngram_range == (1, 2)
    assert clf.min_class_size == 50
    assert clf.is_trained is False


def test_fit_filters_small_classes(train_df):
    """
    Проверяет фильтрацию классов с числом примеров < min_class_size.
    При min_class_size=3 в train_df остаются только 'Образование' и 'Недвижимость'.
    """
    clf = TFIDFClassifier(min_class_size=3)
    clf.fit(df=train_df)
    assert len(clf.label2id) == 2
    assert set(clf.label2id.keys()) == {"Образование", "Недвижимость"}


def test_save_and_load(train_df):
    """
    Проверяет корректность сохранения и загрузки:
    - все ожидаемые файлы создаются,
    - метаданные, словари и флаг is_trained сохраняются и восстанавливаются.
    """
    clf = TFIDFClassifier(min_class_size=2)
    clf.fit(df=train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)

        files = os.listdir(tmpdir)
        assert "meta.pkl" in files
        assert "vectorizer.pkl" in files
        assert "classifier.pkl" in files
        assert "label_mappings.pkl" in files

        clf2 = TFIDFClassifier.load(tmpdir)

        assert clf2.is_trained is True
        assert clf2.metrics["n_classes"] == clf.metrics["n_classes"]
        assert clf2.label2id == clf.label2id
        assert clf2.id2label == clf.id2label


def test_predict_after_load_consistency(train_df, test_df):
    """
    Проверяет, что предсказания не меняются после save/load:
    модель ведёт себя одинаково до и после сериализации.
    """
    clf = TFIDFClassifier(min_class_size=2)
    clf.fit(df=train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        clf2 = TFIDFClassifier.load(tmpdir)

        X = test_df["text"]
        pred1 = clf.classifier.predict(clf.vectorizer.transform(X))
        pred2 = clf2.classifier.predict(clf2.vectorizer.transform(X))

        assert list(pred1) == list(pred2)


def test_predict_filters_unseen_labels(train_df, caplog):
    """
    Проверяет, что predict():
    - логирует предупреждение при наличии меток, отсутствующих в обученной модели,
    - корректно отбрасывает такие строки.
    """
    clf = TFIDFClassifier(min_class_size=1)
    clf.fit(df=train_df)

    unseen_df = pd.DataFrame({
        "text": ["Что-то непонятное"],
        "subrubric": ["Неизвестная рубрика"]
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        unseen_df.to_csv(f.name, index=False)
        tmp_path = f.name

    try:
        with caplog.at_level("WARNING"):
            clf.predict(tmp_path)
        assert "Удалено 1 строк" in caplog.text
    finally:
        os.unlink(tmp_path)


def test_evaluate_empty_dataset():
    """
    Проверяет устойчивость evaluate() к пустому входному файлу:
    - не падает,
    - возвращает accuracy=0.0 и n_samples=0.
    """
    empty_df = pd.DataFrame({"text": [], "subrubric": []})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        empty_df.to_csv(f.name, index=False)
        tmp = f.name

    try:
        clf = TFIDFClassifier()
        clf.label2id = {"A": 0}
        clf.is_trained = True
        result = clf.evaluate(tmp)
        assert result["n_samples"] == 0
        assert result["accuracy"] == 0.0
    finally:
        os.unlink(tmp)


def test_predict_raises_if_not_trained():
    """
    Проверяет, что predict() выбрасывает RuntimeError с понятным сообщением,
    если модель не была обучена.
    """
    clf = TFIDFClassifier()
    df = pd.DataFrame({"text": ["test"], "subrubric": ["test"]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        tmp = f.name

    try:
        with pytest.raises(RuntimeError, match="Модель не обучена"):
            clf.predict(tmp)
    finally:
        os.unlink(tmp)


@pytest.mark.parametrize("min_size, expected_n_classes", [
    (1, 4),
    (2, 4),
    (3, 2),
    (6, 0),
])
def test_fit_min_class_size_parametrized(train_df, min_size, expected_n_classes):
    """
    Проверяет фильтрацию классов при разных min_class_size.
    При min_size=6: 0 классов - ожидаем ValueError.
    """
    clf = TFIDFClassifier(min_class_size=min_size)
    if expected_n_classes == 0:
        with pytest.raises(ValueError, match="не осталось данных"):
            clf.fit(df=train_df)
    else:
        clf.fit(df=train_df)
        assert len(clf.label2id) == expected_n_classes


def test_label_mappings_consistency_after_load(train_df):
    """
    Проверяет, что после загрузки label2id и id2label остаются взаимно-обратными
    """
    clf = TFIDFClassifier(min_class_size=2)
    clf.fit(df=train_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save(tmpdir)
        clf2 = TFIDFClassifier.load(tmpdir)

        for label, idx in clf2.label2id.items():
            assert clf2.id2label[idx] == label
        for idx, label in clf2.id2label.items():
            assert clf2.label2id[label] == idx