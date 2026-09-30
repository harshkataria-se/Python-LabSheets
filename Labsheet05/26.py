import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [55, 60, 65, 70, 75, 80, 85, 88, 92, 95],
    'Assignment_Score': [40, 45, 50, 55, 60, 65, 72, 78, 85, 90],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Assignment_Score']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_without_scaling = LogisticRegression()
model_without_scaling.fit(X_train, y_train)

y_pred_without = model_without_scaling.predict(X_test)
accuracy_without = accuracy_score(y_test, y_pred_without)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_with_scaling = LogisticRegression()
model_with_scaling.fit(X_train_scaled, y_train)

y_pred_with = model_with_scaling.predict(X_test_scaled)
accuracy_with = accuracy_score(y_test, y_pred_with)

print("Logistic Regression Performance Comparison")
print("------------------------------------------")
print("Accuracy without scaling:", accuracy_without)
print("Accuracy with scaling   :", accuracy_with)