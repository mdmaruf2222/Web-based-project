# -----------------------------
# STUDENT PERFORMANCE PREDICTION
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Load Dataset
data = pd.read_csv("student_performance.csv")
print(data.head())

# 2. Basic Info
print(data.info())
print(data.describe())

# 3. Handling Missing Values
data.fillna(method='ffill', inplace=True)

# 4. Create Target Variable (Average Score)
data["average_score"] = data[["math_score", "reading_score", "writing_score"]].mean(axis=1)

# 5. Encode Categorical Columns
label = LabelEncoder()
for col in data.select_dtypes(include="object"):
    data[col] = label.fit_transform(data[col])

# 6. Correlation Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), annot=False, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

# 7. Features & Target
X = data.drop("average_score", axis=1)
y = data["average_score"]

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train Models

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# 10. Evaluation Function
def evaluate_model(name, y_test, pred):
    print(f"\n{name} Model Performance:")
    print("R² Score:", r2_score(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# Evaluate all models
evaluate_model("Linear Regression", y_test, lr_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Gradient Boosting", y_test, gb_pred)

# 11. Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 12. Plot Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=feature_importance)
plt.title("Important Factors Affecting Student Performance")
plt.show()
