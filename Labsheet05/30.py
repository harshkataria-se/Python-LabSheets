import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)

data = {
    'Tenure': [2, 5, 8, 12, 18, 24, 30, 36, 42, 48,
               3, 7, 15, 20, 28, 32, 40, 45, 50, 54],

    'Monthly_Charges': [90, 85, 80, 75, 70, 65, 60, 55, 50, 45,
                        88, 82, 73, 68, 62, 58, 53, 48, 44, 40],

    'Support_Calls': [6, 5, 5, 4, 3, 3, 2, 2, 1, 1,
                      7, 5, 4, 4, 3, 2, 2, 1, 1, 0],

    'Churn': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

print("Customer Churn Dataset:")
print(df)


X = df[['Tenure', 'Monthly_Charges', 'Support_Calls']]
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\nModel trained successfully.")


y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)
print("ROC-AUC  :", roc_auc)


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Stay', 'Churn']
))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Stay', 'Churn'],
    yticklabels=['Stay', 'Churn']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Customer Churn')
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_probability)

plt.figure(figsize=(6, 4))

plt.plot(
    fpr,
    tpr,
    label='Logistic Regression'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random Classifier'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Prediction')
plt.legend()
plt.grid()
plt.show()


new_customer = [[6, 88, 5]]

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)[0][1]

print("\n========== NEW CUSTOMER ==========")

if prediction[0] == 1:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print("Churn Probability:", probability)