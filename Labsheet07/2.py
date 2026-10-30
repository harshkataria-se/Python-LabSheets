import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Target"] = iris.target

print("Dataset Structure:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())