from abc import ABC, abstractmethod

class BaseTextClassifier(ABC):
    @abstractmethod
    def fit(self, df, csv_path, test_size): ...

    @abstractmethod
    def predict(self, texts): ...

    @abstractmethod
    def save(self, path): ...