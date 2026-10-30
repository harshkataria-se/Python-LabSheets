import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTraining Target Shape:", Y_train.shape)
print("Testing Target Shape:", Y_test.shape)