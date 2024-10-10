from zenml import step
from .utils import LoadData, SplitInFour, SplitInTwo, FeatureEngineering


@step
def data_step(filepath: str):
    load_data = LoadData(filepath)
    df = load_data.get_data()
    split_in_two = SplitInTwo(df)
    X, y = split_in_two.split_data()
    fe = FeatureEngineering(X)
    X = fe.engineer()
    split_in_four = SplitInFour(X, y)
    X_train, X_test, y_train, y_test = split_in_four.split_data()
    
    return X_train, X_test, y_train, y_test