import pandas as pd

data = {
    "Name": ["Amit", "Rahul", "Priya", "Neha", "Karan"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "Salary": [25000, 30000, 22000, 28000, 32000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

numerical_features = df.select_dtypes(include=["int64", "float64"])

print("\nNumerical Features:")
print(numerical_features)

print("\nNumerical Feature Names:")
print(numerical_features.columns.tolist())