import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    "Age": [21, 22, None, 23, 21],
    "Marks": [85, None, 78, 88, 92],
    "Salary": [25000, 30000, 22000, None, 32000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy="mean")

processed_data = imputer.fit_transform(df)

df_processed = pd.DataFrame(
    processed_data,
    columns=df.columns
)

print("\nDataset After Handling Missing Values:")
print(df_processed)

print("\nMissing Values After Preprocessing:")
print(df_processed.isnull().sum())