from abc import ABC, abstractmethod
import logging

class BaseTextClassifier(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


    @abstractmethod
    def fit(self, df, csv_path, test_size): ...

    @abstractmethod
    def predict(self, texts): ...

    @abstractmethod
    def save(self, path): ...