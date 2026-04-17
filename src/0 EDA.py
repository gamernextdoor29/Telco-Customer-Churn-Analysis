import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path = r'C:\Users\Mujeeb Bello\Desktop\DATA ANALYSIS\Project 7\data\WA_Fn-UseC_-Telco-Customer-Churn.csv'
pd.set_option('display.max_columns', None)
df = pd.read_csv(file_path)

# Relationship between Contract and Churn
# Create crosstab (counts)
ct = pd.crosstab(df['Contract'], df['Churn'])

# Plot stacked bar chart
sns.set_style("whitegrid")  # seaborn styling
ct.plot(kind='bar', stacked=True)
plt.title('Churn vs Contract Stacked Bar Chart')
plt.xlabel('Contract')
plt.ylabel('Churn')
plt.show()

# Relationship between tenure and Churn
# Using a Boxplot
sns.boxplot(data=df, x='Churn', y='tenure')

plt.title('Tenure Distribution by Churn')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Tenure')
plt.show()

# Using Histogram
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)

plt.title('Tenure Distribution (Churn vs Non-Churn)')
plt.show()

# # Filter for the "Early Danger Zone" (Tenure <= 5 months)
early_tenure = df[df['tenure'] <= 5]

plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=early_tenure)
plt.title('Monthly Charges vs Churn (First 5 Months Only)')
plt.show()


