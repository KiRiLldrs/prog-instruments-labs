from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

from base import BaseTextClassifier


class TFIDFClassifier(BaseTextClassifier):
    def __init__(self, max_features=10000, ngram_range=(1,2), min_class_size=50):
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features = max_features,
            ngram_range = ngram_range,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.is_trained = False
        self.min_class_size=min_class_size


    def fit(self, df=None, csv_path=None, test_size=0.2, nrows=20000):
        """
        Trains the model
        
        :param self: Description
        :param df: Description
        :param csv_path: Description
        :param test_size: Description
        """
        if df is None and csv_path is None:
            raise ValueError("Укажите df или csv_path")
        if df is None:
            df = pd.read_csv(csv_path)
            nr = nrows if len(df)>nrows else len(df)
            df = pd.read_csv(csv_path, nrows=nr)

        df = df.dropna(subset=['text', 'subrubric']).copy()
        df = df.reset_index(drop=True)

        min_class_size = self.min_class_size
        class_counts = df['subrubric'].value_counts()
        valid_classes = class_counts[class_counts >= min_class_size].index
        df = df[df['subrubric'].isin(valid_classes)].reset_index(drop=True)
        self.logger.info(f"Оставлено {len(df)} строк после фильтрации классов с <{min_class_size} примеров")

        if len(df) == 0:
            raise ValueError(
                f"После фильтрации (min_class_size={min_class_size}) не осталось данных. "
                "Невозможно обучить модель без примеров."
            )

        if 'text' not in df.columns or 'subrubric' not in df.columns:
            raise ValueError(f"Файл должен содержать колонки 'text' и 'subrubric'")
        
        self.logger.info(f'Загружен датасет из {len(df)} строк')
        
        self.label2id = {label: idx for idx, label in enumerate(df["subrubric"].unique())}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        df["subrubric_id"] = df["subrubric"].map(self.label2id)

        value_counts = df["subrubric"].value_counts()
        self.logger.info(f"Распределение меток:\n{value_counts.to_string()}")

        n_classes = len(self.label2id)

        if isinstance(test_size, float):
            n_test_est = int(len(df) * test_size)
            if n_test_est < n_classes:
                new_test_size = max(n_classes, 1) 
                self.logger.warning(
                    f"test_size={test_size}: {n_test_est} < {n_classes} классов. "
                    f"Используем test_size={new_test_size} (абсолютное число)."
                )
                final_test_size = new_test_size
            else:
                final_test_size = test_size
        else:
            final_test_size = test_size
            if final_test_size < n_classes:
                raise ValueError(f"test_size={final_test_size} < {n_classes} классов: невозможно стратифицировать")
            
        if isinstance(final_test_size, int) and final_test_size >= len(df):
            final_test_size = max(1, len(df) // 2)
            self.logger.warning(f"test_size превышает размер данных: установлено {final_test_size}")


        train_df, val_df = train_test_split(
            df,
            test_size=final_test_size,
            stratify=df["subrubric"],
            random_state=42
        )
        self.logger.info(f"Разбивка: train={len(train_df)}, val={len(val_df)}")

        X_train = self.vectorizer.fit_transform(train_df["text"])   #обучает валидатор только на train 
        X_val = self.vectorizer.transform(val_df["text"])           #применяет ту же логику к val

        y_train = train_df["subrubric_id"]
        y_val = val_df["subrubric_id"]

        self.logger.info("Начинается обучение классификатора...")
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        self.logger.info(f"Точность на валидации: {accuracy:.4f}")

        self.logger.info(f"Отчёт по классам: \
                         \n{classification_report(y_val, y_pred, target_names=[self.id2label[i] for i in sorted(self.id2label)])}")

        self.is_trained = True
        self.metrics = {
            'label2id': self.label2id,
            'id2label': self.id2label,
            'accuracy' : accuracy,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'n_classes': len(self.label2id),
            'min_class_size': min_class_size,
            'vocab_size': len(self.vectorizer.vocabulary_),
            'classification_report': classification_report(y_val, y_pred, target_names=[self.id2label[i] for i in sorted(self.id2label)])
        }


    def predict(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.logger.info(f"Загружено {len(df)} строк из {csv_path}")

        df = df.dropna(subset=['text', 'subrubric']).reset_index(drop=True)
        self.logger.info(f"После удаления NaN: {len(df)} строк")

        if not self.is_trained:
            self.logger.error("Попытка предсказать на необученной модели")
            raise RuntimeError("Модель не обучена. Вызовите .fit() сначала.")
        
        df['subrubric_id'] = df['subrubric'].map(self.label2id)

        before = len(df)
        df = df.dropna(subset=['subrubric_id']).copy()
        after = len(df)
        if before != after:
            self.logger.warning(f"Удалено {before - after} строк: метки отсутствуют в обученной модели")
        df['subrubric_id'] = df['subrubric_id'].astype(int)

        if len(df) == 0:
            self.logger.warning("Нет строк с известными метками — предсказание невозможно.")
            for handler in self.logger.handlers:
                handler.flush()
            return

        X = self.vectorizer.transform(df['text'])
        y_true = df['subrubric_id'].values
        y_pred = self.classifier.predict(X)

        accuracy = accuracy_score(y_true, y_pred)

        labels = sorted(self.label2id.values())
        target_names = [self.id2label[i] for i in labels]
        
        report_str = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )

        self.logger.info(f"Точность на {csv_path}: {accuracy:.4f}")
        self.logger.info(f"Отчёт по классам:\n{report_str}")


    def evaluate(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['text', 'subrubric']).reset_index(drop=True)
        df['subrubric_id'] = df['subrubric'].map(self.label2id)
        df = df.dropna(subset=['subrubric_id']).copy()

        if len(df) == 0:
            return {'accuracy': 0.0, 'n_samples': 0, 'n_classes': 0}
        
        df['subrubric_id'] = df['subrubric_id'].astype(int)

        X = self.vectorizer.transform(df['text'])
        y_true = df['subrubric_id'].values
        y_pred = self.classifier.predict(X)

        acc = accuracy_score(y_true, y_pred)
        n_samples = len(df)
        n_classes = len(df['subrubric_id'].unique())

        return {
            'accuracy': acc,
            'n_samples': n_samples,
            'n_classes': n_classes,
            'y_true': y_true,
            'y_pred': y_pred
        }
    

    def save(self, path):
        output_dir = path
        os.makedirs(output_dir, exist_ok=True)

        meta = {
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }

        try:
            with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f)

            # Сохраняем TF-IDF векторизатор в файл (чтобы потом обрабатывать новые тексты так же)
            with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer, f)

            # Сохраняем обученный классификатор
            with open(os.path.join(output_dir, "classifier.pkl"), "wb") as f:
                pickle.dump(self.classifier, f)

            # Сохраняем словари перевода между названиями категорий и числами
            with open(os.path.join(output_dir, "label_mappings.pkl"), "wb") as f:
                pickle.dump({"label2id": self.label2id, "id2label": self.id2label}, f)
        except Exception as e:
            self.logger.exception("Ошибка при сохранении модели")
            raise
    
    
    @classmethod
    def load(cls, path):
        try:
            with open(Path(path,"meta.pkl"), "rb") as f:
                meta = pickle.load(f)

            obj = cls()
            obj.is_trained = meta["is_trained"]
            obj.metrics = meta["metrics"]
            obj.label2id = meta["metrics"]["label2id"]
            obj.id2label = meta["metrics"]["id2label"]

            with open(Path(path, "vectorizer.pkl"), "rb") as f:
                obj.vectorizer = pickle.load(f)

            with open(Path(path, "classifier.pkl"), "rb") as f:
                obj.classifier = pickle.load(f)

            with open(Path(path, "label_mappings.pkl"), "rb") as f:
                mappings = pickle.load(f)
                obj.id2label = mappings["id2label"]

            obj.logger.info("Модель загружена")
        except FileNotFoundError as e:
            obj.logger.error("Файл не найден")
            raise
        except Exception as e:
            obj.logger.exception("Ошибка при загрузке модели")

        return obj


        





    
