
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

cv = KFold(n_splits=5, shuffle=True, random_state=42)

metrics = ['euclidean', 'manhattan']

for metric in metrics:
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric
    )

    scores = cross_val_score(
        knn, X, y, cv=cv, scoring='accuracy'
    )

    print(f"\nDistance Metric: {metric}")
    print("Fold Accuracies:", scores)
    print("Mean Accuracy:", round(scores.mean(), 4))

