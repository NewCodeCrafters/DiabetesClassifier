from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ct = ColumnTransformer([
    ('encoding', OneHotEncoder(), ['gender']),
    ('scaling', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
])