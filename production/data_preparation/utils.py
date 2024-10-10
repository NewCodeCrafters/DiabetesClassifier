import logging
import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Annotated, Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler



logger = logging.getLogger(__name__)

class LoadData:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
    
    def get_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.filepath)
            return data
        except FileNotFoundError as e:
            logging.info(f'Error -> {e}')
            
            
class SplitData(ABC):
    
    @abstractmethod
    def split_data(self):
        pass

class SplitInTwo(SplitData):
    def __init__(self, data: pd.Dataframe) -> None:
        self.data = data
        
    def split_data(self) -> Tuple[
        Annotated[pd.DataFrame, 'X'],
        Annotated[pd.Series, 'y']
    ]:
        try:
            X = self.data.drop('diabetes', axis=1).copy()
            y = self.data.loc[:, 'diabetes'].copy()
        except Exception as e:
            logging.info(f'Error -> {e}')
        
        return X, y
    
    
class SplitInFour(SplitData):
    def __init__(self, X: np.array, y: pd.Series) -> None:
        self.X = X
        self.y = y
        
    def split_data(self) -> Tuple[
        Annotated[pd.DataFrame, 'X_train'],
        Annotated[pd.DataFrame, 'X_test'],
        Annotated[pd.DataFrame, 'y_train'],
        Annotated[pd.DataFrame, 'y_test']
    ]:
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=42, test_size=.2)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.info(f'Error -> {e}')
    
class FeatureEngineering:
    def __init__(self, X) -> None:
        self.X = X
        
    def preprocess(self) -> ColumnTransformer:
        ct = ColumnTransformer([
            ('encoding', OneHotEncoder(), ['gender']),
            ('scaling', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
        ])
        return ct
    
    def engineer(self) -> np.array:
        ct = self.preprocess()
        preprocessed_X = ct.fit_transform(self.X)
        return preprocessed_X

        
        