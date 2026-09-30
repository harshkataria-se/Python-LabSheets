import pandas as pd

data = {
    'Study_Hours': [2, 3, 4, None, 6, 7, 8, 9, 10, 11],
    'Attendance': [55, 60, None, 70, 75, 80, 85, 88, 92, 95],
    'Assignment_Score': [40, 45, 50, 55, None, 65, 72, 78, 85, 90],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

print("Dataset Before Handling Missing Values:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].mean())
df['Attendance'] = df['Attendance'].fillna(df['Attendance'].mean())
df['Assignment_Score'] = df['Assignment_Score'].fillna(
    df['Assignment_Score'].mean()
)
print("\nDataset After Handling Missing Values:")
print(df)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())