import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [55, 60, 65, 70, 75, 80, 85, 88, 92, 95],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]

y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Results:   ", y_test.values)
print("Predicted Results:", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

study_hours = 6
attendance = 78

new_student = [[study_hours, attendance]]
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\nNew Student Result: Pass")
else:
    print("\nNew Student Result: Fail")