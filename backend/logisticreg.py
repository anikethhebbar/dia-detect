import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
# Reading the updated dataset
dataset = pd.read_csv("data.csv")
# Feature Selection: Selecting all features except the target 'OutcomeBinary'
features = dataset.drop(columns='OutcomeBinary')
target = dataset['OutcomeBinary']
# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Model Training
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
# Predictions
y_pred = log_reg.predict(X_test_scaled)
# Performance Metrics
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:\n", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Save the model using pickle (optional)
import pickle
pickle.dump(log_reg, open('logistic_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))