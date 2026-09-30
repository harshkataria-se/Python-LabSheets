import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'Study_Hours': [2, 3, 4, 5, 6],
    'Attendance': [60, 70, 75, 85, 90],
    'Assignment_Score': [50, 60, 70, 80, 95]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nStandardized Dataset:")
print(scaled_df)

print("\nMean after Standardization:")
print(scaled_df.mean())

print("\nStandard Deviation after Standardization:")
print(scaled_df.std())