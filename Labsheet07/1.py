import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Target"] = iris.target

print("Classification Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nTarget Classes:", iris.target_names)