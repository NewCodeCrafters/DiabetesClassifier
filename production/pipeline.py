from zenml import pipeline
from data_step.steps import data_step, preprocessing, split_into_train_and_test
from model_step.steps import training, evaluation, prediction

@pipeline()
def diabetes_classifier_pipeline():
    X, y = data_step('../data/best_df.csv')
    preprocessed_X = preprocessing(X)
    X_train, X_test, y_train, y_test = split_into_train_and_test(preprocessed_X, y)
    model = training('tree', X_train, y_train)
    pred = prediction(model, X_test)
    accuracy, precision, recall, f1 = evaluation(pred, y_test)