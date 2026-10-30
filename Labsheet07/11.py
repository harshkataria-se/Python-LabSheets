import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(criterion="gini", random_state=42)
model.fit(X, y)

importance = model.feature_importances_

for name, value in zip(iris.feature_names, importance):
    print(name, ":", value)

plt.barh(iris.feature_names, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")
plt.show()