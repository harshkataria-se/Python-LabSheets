import pandas as pd
import numpy as np

data = {
    "Message": [
        "Win a free prize",
        "Call me tomorrow",
        np.nan,
        "Get a free gift",
        "Meeting at 10 AM"
    ],
    "Length": [16, 17, 25, np.nan, 17],
    "Category": ["Spam", "Ham", "Spam", np.nan, "Ham"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

df["Message"] = df["Message"].fillna("Unknown")
df["Length"] = df["Length"].fillna(df["Length"].mean())
df["Category"] = df["Category"].fillna(df["Category"].mode()[0])

print("\nDataset After Preprocessing:")
print(df)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())