import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Tenure': [2, 5, 8, 12, 18, 24, 30, 36, 42, 48],
    'Monthly_Charges': [90, 85, 80, 75, 70, 65, 60, 55, 50, 45],
    'Support_Calls': [6, 5, 5, 4, 3, 3, 2, 2, 1, 1],
    'Churn': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

X = df[['Tenure', 'Monthly_Charges', 'Support_Calls']]

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Actual Churn:   ", y_test.values)
print("Predicted Churn:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_customer = [[6, 88, 5]]

new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)

if prediction[0] == 1:
    print("New Customer: Likely to Churn")
else:
    print("New Customer: Likely to Stay")