import pandas as pd

data = {
    'Study_Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [55, 60, 65, 70, 75, 80, 85, 88, 92, 95],
    'Assignment_Score': [40, 45, 50, 55, 60, 65, 72, 78, 85, 90],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Assignment_Score']]
Y = df['Result']

print("Independent Variables (X):")
print(X)

print("\nDependent Variable (Y):")
print(Y)