
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
X = iris.data
y = iris.target

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

k_values = [1, 3, 5, 7, 9]
best_k = 0
best_accuracy = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn, X, y, cv=kfold, scoring='accuracy'
    )

    mean_accuracy = scores.mean()

    print(f"k = {k}, Mean Accuracy = {mean_accuracy:.2f}")

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

print("\nBest k value:", best_k)
print("Best Cross-Validation Accuracy:", round(best_accuracy, 2))

