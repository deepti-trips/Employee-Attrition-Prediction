import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
project_path = os.getcwd()
print("Project Path = ",project_path)

file_path = project_path + "\\dataset.xlsx"
print("File Path = ",file_path)

df = pd.read_excel(file_path)
print(df)

# Convert categorical data
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df[['Age','MonthlyIncome','YearsAtCompany','JobSatisfaction']]
y = df['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)
print("Model Accuracy:", acc)

# Save model
pickle.dump(model, open("attrition_model.pkl", "wb"))

import pickle

pickle.dump(model, open("attrition_model.pkl", "wb"))

import pickle
import numpy as np

model = pickle.load(open("attrition_model.pkl", "rb"))

# Example employee data
sample = np.array([[35, 5000, 5, 3]])

prediction = model.predict(sample)

if prediction == 1:
    print("Employee likely to leave")
else:
    print("Employee likely to stay")