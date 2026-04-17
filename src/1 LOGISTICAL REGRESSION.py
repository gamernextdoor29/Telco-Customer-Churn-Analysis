# CODE 1
# USING ONLY LOGISTICAL REGRESSION

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report


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
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_predict = pipe.predict(X_test)
y_pred_proba = pipe.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_pred_proba)

print("Test ROC-AUC:", roc_score)

print(classification_report(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))

# # Filter for the "Early Danger Zone" (Tenure <= 5 months)
# early_tenure = df[df['tenure'] <= 5]
#
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='Churn', y='MonthlyCharges', data=early_tenure)
# plt.title('Monthly Charges vs Churn (First 5 Months Only)')
# plt.show()



# X = df.drop('charges', axis = 1)
# y = df['charges']
#
# # Create a column for 'Obese Smokers', 'healthy non smokers', 'healthy smokers' and 'obese non smokers'
# X['obese_smoker'] = ((X['bmi'] > 30) & (X['smoker'] == 'yes')).astype(int)
# X['healthy_non_smoker'] = ((X['bmi'] < 30) & (X['smoker'] == 'no')).astype(int)
# X['healthy_smoker'] = ((X['bmi'] < 30) & (X['smoker'] == 'yes')).astype(int)
# X['obese_non_smoker'] = ((X['bmi'] > 30) & (X['smoker'] == 'no')).astype(int)
#
#
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# X_train = pd.get_dummies(X_train_raw)
# X_test = pd.get_dummies(X_test_raw)
# X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
#
# y_train_log = np.log1p(y_train)
#
# # Define which columns are which
# # num_features = ['age', 'bmi', 'children']
# # cat_features = ['sex', 'smoker', 'region']
#
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('num', StandardScaler(), num_features),
# #         ('cat', OneHotEncoder(drop='first'), cat_features)
# #     ]
# # )
# rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#
# pipe = Pipeline([
#     ('preprocessing', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=False)),
#     #('selector', RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)),
#     #('selector', RFECV(estimator=LinearRegression(), step=1, cv=5, scoring='r2')),
#     ('model', LinearRegression())
# ])
#
# pipe.fit(X_train, y_train_log)
# prediction_log = pipe.predict(X_test)
# final_predictions = np.expm1(prediction_log)
#
# # for R_SQUARED
# r2 = r2_score(y_test, final_predictions)
# # for MAE
# mae = mean_absolute_error(y_test, final_predictions)
# # for RMSE
# rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
# # CHECKING FOR UNDERFITTING AND OVERFITTING
# train_prediction = np.expm1(pipe.predict(X_train))
# print(f'Train MAE:{mean_absolute_error(y_train, train_prediction)}')
# print(f'Test MAE: {mean_absolute_error(y_test, final_predictions)}')
#
#
# print(f"R Squared: ${r2:,.2f}")
# print(f"Mean Absolute Error: ${mae:,.2f}")
# print(f"Root Mean Squared Error: ${rmse:,.2f}")
#
# avg = df['charges'].mean()
# per = mae/avg
# print(per * 100)
# print('code 1')
#
# #VISUALIZATION
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=final_predictions, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
# plt.xlabel('Actual Charges ($)')
# plt.ylabel('Predicted Charges ($)')
# plt.title('Actual vs. Predicted Charges')
# plt.show()
#
# # ploting histogram of residuals
# residuals = y_test - final_predictions
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True)
# plt.title('Distribution of Prediction Errors (Residuals)')
# plt.xlabel('Error ($)')
# plt.show()




