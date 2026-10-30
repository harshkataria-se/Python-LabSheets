import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    response_method="predict",
    cmap="viridis",
    alpha=0.4,
    xlabel=iris.feature_names[2],
    ylabel=iris.feature_names[3]
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black",
    cmap="viridis"
)

plt.title("Random Forest Decision Boundary")
plt.show()