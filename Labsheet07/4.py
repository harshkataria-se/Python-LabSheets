import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    "Age": [20, 25, None, 30, 22],
    "Gender": ["Male", "Female", "Male", None, "Female"],
    "City": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"]
})

print("Original Dataset:")
print(df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["City"] = le.fit_transform(df["City"])

print("\nPreprocessed Dataset:")
print(df)