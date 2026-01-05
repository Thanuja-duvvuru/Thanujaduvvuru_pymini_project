# DATA LOADING
#Load dataset using pandas and seaborn
import pandas as pd
import seaborn as sns
df=pd.read_csv("Salary_Data.csv")
#Inspect the first few rows with df.head()
print("First 5 rows of dataset:\n",df.head())
#Check shape and column types with df.info()
print("\nDataset Info:")
print(df.info())
#DATA CLEANING
#Checking missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
#EXPLORATORY DATA ANALYSIS (EDA)
print("\nSummary Statistics:")
print(df.describe())
#DATA VISUALIZATION
# importing matplot library for graphs
import matplotlib.pyplot as plt
#Univariate Analysis
df.hist(figsize=(6,4))
plt.title("Histogram on salary dataset")
plt.show()
#Bivariate Analysis
plt.scatter(df['Years of Experience'], df['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Years of Experience")
plt.show()
#Correlation Heatmap
numeric_df=df.select_dtypes(include='number')
corr=numeric_df.corr()
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
#PREDICTIVE MODELING
#LINEAR REGRESSION
#Regression Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
# Features and target
X = df[['Years of Experience']]   # Independent variable
y = df['Salary']              # Dependent variable
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# Predictions
y_pred = lin_reg.predict(X_test)
# Evaluation
print("Linear Regression Results")
print("Intercept:", lin_reg.intercept_)
print("Coefficient:", lin_reg.coef_)
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:",np.sqrt(mse) )
print("R² Score:",r2_score(y_test, y_pred))
