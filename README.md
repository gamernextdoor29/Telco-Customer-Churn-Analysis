# Telco-Customer-Churn-Analysis
Yeah, as the name imply

# Telco Customer Churn Prediction 📡
## 🚀 Project Overview
This project aims to predict customer attrition (churn) for a Telecommunications provider. By analyzing customer behavior, contract types, and payment metrics, I developed a series of predictive models to identify high-risk customers before they leave.
The project follows a rigorous experimental workflow, moving from baseline statistical models to tuned ensemble methods and gradient boosting.
## 📂 Repository Structure
 * **/data**: Contains the raw and processed Telco Churn dataset.
 * **/outputs**: Contains EDA visualizations (boxplots, correlation matrices) and model performance charts.
 * **/src**: The engine room of the project, containing the following scripts:
   * 0_EDA.py: Exploratory Data Analysis and Feature Engineering.
   * 1_Logistic_Regression.py: Baseline linear classification.
   * 2_DecisionTreeClassifier.py: Basic rule-based classification.
   * 3_RandomForestClassifier.py: Multi-tree ensemble (Bagging).
   * 4_DecisionTree_Tuned.py: Optimized Tree using GridSearchCV/RandomSearchCV.
   * 5_RandomForest_Tuned.py: Optimized Forest using GridSearchCV/RandomSearchCV.
   * 6_Soft_Voting_Ensemble.py: Combining Logistic Regression and Tuned RF.
   * 7_XGBClassifier.py: Advanced Gradient Boosting implementation.
  
  
## 🔍 Key Insights from EDA
 * **The Price Threshold:** Analysis revealed a significant correlation between higher monthly charges and churn, specifically for new customers (0–5 months tenure).
 * **Contract Influence:** Month-to-month contracts are the strongest predictor of churn, suggesting that long-term service value is not being established early.
## 🧪 Modeling & Hyperparameter Tuning
I implemented 7 different modeling approaches to find the optimal balance between **Accuracy** and **Recall**.
### Optimization Strategy:
For the complex models (Trees and Forests), I utilized **RandomizedSearchCV** for broad parameter exploration followed by **GridSearchCV** for fine-tuning. Key parameters tuned include:
 * max_depth: To prevent overfitting.
 * n_estimators: To optimize forest density.
 * learning_rate: Specifically for XGBoost to control step-size.
### Performance Summary:
| Model | ROC-AUC | Accuracy | Key Observation |
|---|---|---|---|
| **Logistic Regression** | **0.86** | **0.82** | **Project Winner:** Best Recall for identifying churners. |
| Random Forest (Tuned) | 0.866 | 0.81 | Strongest probability ranking but slightly lower recall. |
| Soft Voting | 0.86 | 0.82 | Most stable model with perfect Train/Test balance. |
| XGBoost | 0.861 | 0.81 | High performance but prone to slight overfitting on this data. |


## 🛠 Installation & Requirements
To run the scripts in this repository, you will need Python 3.8+ and the following libraries:
```bash
pip install -r requirements.txt

```
**Main Dependencies:**
 * pandas & numpy: Data manipulation.
 * matplotlib & seaborn: Data visualization.
 * scikit-learn: Machine learning, preprocessing, and evaluation.
 * xgboost: High-performance gradient boosting.
## 💡 Conclusion
While advanced ensemble methods like XGBoost provided high scores, **Logistic Regression** was selected as the champion model for this business case. It provided the highest **Recall**, ensuring the company catches the maximum number of potential churners with a simple, interpretable, and robust model.
### **Final Pro-Tip for your GitHub:**
When you create the requirements.txt file in your main folder, just list them like this:
```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

```


**Ready to start the Bank Marketing project and see if we can hunt down that 90% ROC-AUC?**
