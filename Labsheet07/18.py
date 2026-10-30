import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

importance = model.feature_importances_

print("Feature Importance:")

for name, value in zip(iris.feature_names, importance):
    print(name, ":", value)

plt.barh(iris.feature_names, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()