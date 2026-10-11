import pandas as pd

dataset = pd.read_csv("classification_data.csv")

X = dataset.drop("Species", axis=1)
Y = dataset["Species"]

print("Independent Variables (X):")
print(X)

print("\nDependent Variable (Y):")
print(Y)