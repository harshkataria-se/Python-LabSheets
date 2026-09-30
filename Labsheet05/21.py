import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

probabilities = model.predict_proba(X_test_scaled)

print("Predicted Class Probabilities:")
print(probabilities)

sample_numbers = range(1, len(X_test) + 1)

plt.figure(figsize=(7, 4))

plt.bar(
    [x - 0.2 for x in sample_numbers],
    probabilities[:, 0],
    width=0.4,
    label='Fail Probability'
)

plt.bar(
    [x + 0.2 for x in sample_numbers],
    probabilities[:, 1],
    width=0.4,
    label='Pass Probability'
)

plt.xlabel('Test Sample')
plt.ylabel('Probability')
plt.title('Predicted Class Probabilities')
plt.xticks(list(sample_numbers))
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y')

plt.show()