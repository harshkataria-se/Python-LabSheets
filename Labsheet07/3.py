import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Target"] = iris.target

X = df.drop("Target", axis=1)
Y = df["Target"]

print("Independent Variables (X):")
print(X.head())

print("\nDependent Variable (Y):")
print(Y.head())