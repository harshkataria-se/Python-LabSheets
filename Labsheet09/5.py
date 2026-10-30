import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Message_Length": [25, 40, 15, 60, 35],
    "Word_Count": [5, 8, 3, 12, 7],
    "Special_Characters": [2, 4, 1, 6, 3]
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

print("\nMean of Standardized Features:")
print(scaled_df.mean())

print("\nStandard Deviation of Standardized Features:")
print(scaled_df.std(ddof=0))