from zenml import step
import numpy as np
import pandas as pd
from typing import Tuple, Annotated
from .func import LoadData, SplitInFour, SplitInTwo, FeatureEngineering
from .utils import ct


@step(enable_cache=False)
def data_step(filepath: str) -> Tuple[
    Annotated[pd.DataFrame, 'X'],
    Annotated[pd.Series, 'y']
]:
    load_data = LoadData(filepath)
    df = load_data.get_data()
    split_in_two = SplitInTwo(df)
    X, y = split_in_two.split_data()
    
    return X, y
    
    
@step
def preprocessing(X: pd.DataFrame) -> np.ndarray:
    fe = FeatureEngineering(X)
    X = fe.engineer(ct)
    
    return X


@step
def split_into_train_and_test(X: np.ndarray, y: pd.Series) -> Tuple[
        Annotated[np.ndarray, 'X_train'],
        Annotated[np.ndarray, 'X_test'],
        Annotated[pd.Series, 'y_train'],
        Annotated[pd.Series, 'y_test']
    ]:
    split_in_four = SplitInFour(X, y)
    X_train, X_test, y_train, y_test = split_in_four.split_data()
    
    return X_train, X_test, y_train, y_test
