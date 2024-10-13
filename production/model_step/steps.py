from zenml import step
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Annotated
from sklearn.base import ClassifierMixin
from .functions import Models, Predictor, Evaluator


@step
def training(model_name: str, X_train: np.ndarray, y_train: pd.Series) -> ClassifierMixin:
    model = Models(model_name)
    trained_model = model.train_model(X_train, y_train)
    return trained_model

@step
def prediction(model: ClassifierMixin, X_test: np.ndarray) -> np.ndarray:
    predictor = Predictor(model, X_test)
    y_pred = predictor.predict()
    return y_pred

@step
def evaluation(y_pred: np.ndarray, y_test: pd.Series) -> Tuple[
        Annotated[float, 'accuracy'],
        Annotated[float, 'recall'],
        Annotated[float, 'precision'],
        Annotated[float, 'f1']
    ]:
    evaluator = Evaluator(y_pred, y_test)
    accuracy, precision, recall, f1 = evaluator.evaluate()
    return accuracy, precision, recall, f1
