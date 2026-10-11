import pandas as pd

dataset = pd.read_csv("classification_data.csv")

print("Missing Values Before Preprocessing:")
print(dataset.isnull().sum())

numerical_columns = dataset.select_dtypes(include="number").columns
dataset[numerical_columns] = dataset[numerical_columns].fillna(
    dataset[numerical_columns].mean()
)

categorical_columns = dataset.select_dtypes(include="object").columns
for column in categorical_columns:
    dataset[column] = dataset[column].fillna(dataset[column].mode()[0])

dataset.to_csv("classification_data.csv", index=False)

print("\nMissing Values After Preprocessing:")
print(dataset.isnull().sum())

print("\nPreprocessed Dataset:")
print(dataset)