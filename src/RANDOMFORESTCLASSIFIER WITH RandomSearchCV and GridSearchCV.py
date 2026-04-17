# CODE 5
# USING RANDOMFORESTCLASSIFIER WITH RandomSearchCV and GridSearchCV

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
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

pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])


from scipy.stats import randint


# ==============================
# 1. RandomizedSearchCV (Exploration)
# ==============================
# Random Search space, using randint to generate random number in this ranges and RandomSearchCV will break it down

param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': randint(5, 50),
    'model__min_samples_split': randint(2, 20),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['sqrt', 'log2', None],
    'model__bootstrap': [True, False]
}

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best Random Params:", random_search.best_params_)
print("Best Random Score:", random_search.best_score_)



# ==============================
# 2. Build Grid Around Best Params (Refinement)
# ==============================
# now that RandomSearchCV has narrowed it down let's use GridSearchCV to get the best possible parameter

best = random_search.best_params_

param_grid = {
    'model__n_estimators': [
        best['model__n_estimators'] - 50,
        best['model__n_estimators'],
        best['model__n_estimators'] + 50
    ],
    'model__max_depth': [
        best['model__max_depth'] - 5,
        best['model__max_depth'],
        best['model__max_depth'] + 5
    ],
    'model__min_samples_split': [
        max(2, best['model__min_samples_split'] - 2),
        best['model__min_samples_split'],
        best['model__min_samples_split'] + 2
    ],
    'model__min_samples_leaf': [
        max(1, best['model__min_samples_leaf'] - 1),
        best['model__min_samples_leaf'],
        best['model__min_samples_leaf'] + 1
    ],
    'model__max_features': [best['model__max_features']],
    'model__bootstrap': [best['model__bootstrap']]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("Best Grid Params:", grid_search.best_params_)
print("Best Grid Score:", grid_search.best_score_)


# ==============================
# 3. Final Model Evaluation
# ==============================
best_model = grid_search.best_estimator_

y_predict = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_pred_proba)

print("Test ROC-AUC:", roc_score)

print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
