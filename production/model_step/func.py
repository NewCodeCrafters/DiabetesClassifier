from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Annotated, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier


class Models:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        
    def train_model(self, X_train: np.ndarray, y_train: pd.Series) -> ClassifierMixin:
        model_instances = {
            'rf': RandomForestClassifier(random_state=42), 
            'tree': DecisionTreeClassifier(random_state=42),
            'xgb': XGBClassifier(random_state=42),
        }
        
        try:
            model = model_instances[self.model_name]
            model.fit(X_train, y_train)
        except Exception as e:
            logging.info(f'Error in selecting Models -> {e}')
        
        
        return model
    
    
class Predictor:
    def __init__(self, model: ClassifierMixin, X_test: np.ndarray) -> None:
        self.model = model
        self.X_test = X_test
        
    def predict(self) -> np.ndarray:
        try:
            y_pred = self.model.predict(self.X_test)
            return y_pred
        except Exception as e:
            logging.info(f'Error -> {e}')   

    
class Evaluator:
    def __init__(self, y_pred: np.ndarray, y_test: pd.Series) -> None:
        self.y_pred = y_pred
        self.y_test = y_test
        
    @staticmethod
    def measure_model(y_test: pd.Series, y_pred: np.ndarray) -> Dict:
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            'accuracy': acc,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }
        
    def evaluate(self) -> Tuple[
        Annotated[float, 'accuracy'],
        Annotated[float, 'recall'],
        Annotated[float, 'precision'],
        Annotated[float, 'f1']
    ]:
        try:
            metrics = self.measure_model(y_test=self.y_test, y_pred=self.y_pred)
            return tuple(metrics.values())
        except Exception as e:
            logging.info(f'Error -> {e}')
        