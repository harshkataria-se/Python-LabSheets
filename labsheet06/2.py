import pandas as pd

dataset = pd.read_csv("classification_data.csv")

print("----- Dataset Structure -----")
print("Shape:", dataset.shape)
print("\nColumn Names:")
print(dataset.columns)
print("\nData Types:")
print(dataset.dtypes)

print("\nDataset Information:")
dataset.info()

print("\n----- Statistical Summary -----")
print(dataset.describe())

print("\n----- Categorical Summary -----")
print(dataset.describe(include="object"))