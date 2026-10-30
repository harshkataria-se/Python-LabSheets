import pandas as pd

data = {
    "Message_Length": [25, 40, 15, 60, 35],
    "Word_Count": [5, 8, 3, 12, 7],
    "Special_Characters": [2, 4, 1, 6, 3],
    "Category": ["Spam", "Ham", "Spam", "Ham", "Spam"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

numerical_features = df.select_dtypes(include=["int64", "float64"])

print("\nSelected Numerical Features:")
print(numerical_features)

print("\nFeature Names:")
print(numerical_features.columns.tolist())

print("\nShape of Selected Features:")
print(numerical_features.shape)