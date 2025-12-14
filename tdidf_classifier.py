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
    def __init__(self, max_features=10000, ngram_range=(1,2)):
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


    def fit(self, df=None, csv_path=None, test_size=0.2):
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

        print(f'Загружено {len(df)} исходных примеров')

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Файл должен содержать колонки 'text' и 'label'")
        
        self.label2id = {label: idx for idx, label in enumerate(df["label"].unique())}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        df["label_id"] = df["label"].map(self.label2id)

        print("Распределение категорий:")
        print(df["label"].value_counts())

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=42
        )

        X_train = self.vectorizer.fit_transform(train_df["text"])   #обучает валидатор только на train 
        X_val = self.vectorizer.transform(val_df["text"])           #применяет ту же логику к val

        y_train = train_df["label_id"]
        y_val = val_df["label_id"]

        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        print(f"Точность на валидации: {accuracy:.4f}")

        print("\nОтчёт по классам:")
        print(classification_report(y_val, y_pred, target_names=[self.id2label[i] for i in sorted(self.id2label)]))

        self.is_trained = True
        self.metrics = {
            'accuracy' : accuracy,
            'classification_report': classification_report(y_val, y_pred, target_names=[self.id2label[i] for i in sorted(self.id2label)])
        }


    def predict(self, text: str)->str:
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Вызовите .fit() сначала.")

        X = self.vectorizer.transform([text])
        pred_id = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]

        confidence = float(probabilities[pred_id])
        category = self.id2label[pred_id]

        return category, confidence
    

    def save(self, path):
        output_dir = path
        os.makedirs(output_dir, exist_ok=True)

        meta = {
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }
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
    
    
    @classmethod
    def load(cls, path):
        with open(Path(path,"meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        obj = cls()
        obj.is_trained = meta["is_trained"]
        obj.metrics = meta["metrics"]

        with open(Path(path, "vectorizer.pkl"), "rb") as f:
            obj.vectorizer = pickle.load(f)

        with open(Path(path, "classifier.pkl"), "rb") as f:
            obj.classifier = pickle.load(f)

        with open(Path(path, "label_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
            obj.id2label = mappings["id2label"]

        print("Модель загружена")

        return obj


        





    
