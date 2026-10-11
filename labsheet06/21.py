
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=3)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    knn, X, y, cv=kfold, scoring='accuracy'
)

print("Accuracy for each fold:")
for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.2f}")

print("\nMean Accuracy:", round(scores.mean(), 2))
