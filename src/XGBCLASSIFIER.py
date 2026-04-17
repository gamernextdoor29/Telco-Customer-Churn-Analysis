# CODE 7
# USING XGBCLASSIFIER
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metjrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

file_path = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 7\WA_Fn-UseC_-Telco-Customer-Churn.csv'
pd.set_option('display.max_columns', None)
df = pd.read_csv(file_path)

df['Churn_coded'] = df['Churn'].str.lower().map({'yes':1, 'no':0})
col_to_drop = ['Churn', 'customerID']
df = df.drop(col_to_drop, axis = 1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)


X = df.drop('Churn_coded', axis=1)
y = df['Churn_coded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_columns = X_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()
cat_columns = X_train.select_dtypes(include = ['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('nums', StandardScaler(), num_columns),
    ('cats', OneHotEncoder(), cat_columns)
])

from xgboost import XGBClassifier

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5))
])

xgb_pipe.fit(X_train, y_train)
xgb_pipe.fit(X_train, y_train)

y_predict = xgb_pipe.predict(X_test)
y_pred_proba = xgb_pipe.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_pred_proba)

print("Test ROC-AUC:", roc_score)

print(classification_report(y_test, y_predict))


y_train_predict = xgb_pipe.predict(X_train)

print('CLASSIFICATION REPORT FOR TRAINING')
print(classification_report(y_train, y_train_predict))


print(confusion_matrix(y_test, y_predict))
