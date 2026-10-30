import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Predicted Class Names:", iris.target_names[y_pred])