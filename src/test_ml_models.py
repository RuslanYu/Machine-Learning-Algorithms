# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:37:01 2025

@author: RuslanYushvaev
"""

"""
test_ml_models.py

Demonstrates how to import and use the classes in ml_models.py.
Tests the LinearRegressionGD model on three real-world datasets:
1) Diabetes
2) California Housing
3) Auto MPG
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes, fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt

# Import our ML code
from ml_models import LinearRegressionGD

#==========================================================
# Example 1: Diabetes Dataset
#==========================================================
print("=== Example 1: Diabetes Dataset ===")
diabetes = load_diabetes()
X_d = diabetes.data  # shape: (442, 10)
y_d = diabetes.target

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

# Create and fit our model
model_diabetes = LinearRegressionGD(learning_rate=0.01, epochs=1000)
model_diabetes.fit(X_d_train, y_d_train)

# Predict
y_d_pred = model_diabetes.predict(X_d_test)

# Evaluate
mse_diabetes = mean_squared_error(y_d_test, y_d_pred)
print("Diabetes Test MSE =", mse_diabetes)
print("Learned beta (first 5):", model_diabetes.beta[:5], "...")


#==========================================================
# Example 2: California Housing Dataset
#==========================================================
print("\n=== Example 2: California Housing Dataset ===")
cal = fetch_california_housing()
X_c = cal.data       # shape: (20640, 8)
y_c = cal.target

X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42
)

# Because scales vary drastically, let's do a quick standardization
X_c_mean = X_c_train.mean(axis=0)
X_c_std = X_c_train.std(axis=0)
X_c_std[X_c_std == 0] = 1.0

X_c_train_scaled = (X_c_train - X_c_mean) / X_c_std
X_c_test_scaled  = (X_c_test - X_c_mean) / X_c_std

model_cal = LinearRegressionGD(learning_rate=0.000001, epochs=1000)
model_cal.fit(X_c_train_scaled, y_c_train)
y_c_pred = model_cal.predict(X_c_test_scaled)

mse_cal = mean_squared_error(y_c_test, y_c_pred)
print("California Housing Test MSE =", mse_cal)
print("Learned beta (first 5):", model_cal.beta[:5], "...")


#==========================================================
# Example 3: Auto MPG Dataset
#==========================================================
print("\n=== Example 3: Auto MPG Dataset ===")
mpg_data = sns.load_dataset("mpg").dropna()
# Numeric columns for features
feature_cols = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
X_m = mpg_data[feature_cols]
y_m = mpg_data["mpg"]

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_m, y_m, test_size=0.2, random_state=42
)

# Optional scale
X_m_mean = X_m_train.mean(axis=0)
X_m_std  = X_m_train.std(axis=0)
X_m_std[X_m_std == 0] = 1.0

X_m_train_scaled = (X_m_train - X_m_mean) / X_m_std
X_m_test_scaled  = (X_m_test  - X_m_mean) / X_m_std

model_mpg = LinearRegressionGD(learning_rate=0.0001, epochs=3000)
model_mpg.fit(X_m_train_scaled, y_m_train)

y_m_pred = model_mpg.predict(X_m_test_scaled)
mse_mpg = mean_squared_error(y_m_test, y_m_pred)
print("Auto MPG Test MSE =", mse_mpg)
print("Learned beta (first 5):", model_mpg.beta[:5], "...")

#---------------------------------------------
# Optional: Plot Cost History for Auto MPG
#---------------------------------------------
plt.figure()
plt.plot(model_mpg.cost_history, color='purple')
plt.title("Gradient Descent Cost Over Iterations - Auto MPG")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.show()
