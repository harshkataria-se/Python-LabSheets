import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "Salary": [25000, 30000, 22000, 28000, 32000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("\nStandardized Dataset:")
print(df_scaled)

print("\nMean of Standardized Features:")
print(df_scaled.mean())

print("\nStandard Deviation of Standardized Features:")
print(df_scaled.std(ddof=0))