import pandas as pd
from sklearn.model_selection import train_test_split

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
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

print("\nTraining Target:")
print(y_train)

print("\nTesting Target:")
print(y_test)

print("\nTraining Dataset Size:", len(X_train))
print("Testing Dataset Size:", len(X_test))