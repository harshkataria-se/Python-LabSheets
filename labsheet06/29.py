
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

parameters = {
    "knn__n_neighbors": [1, 3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

grid = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:",
      round(grid.best_score_ * 100, 2), "%")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nPredicted Labels:", y_pred)
print("Actual Labels:   ", y_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("kNN Classification - Confusion Matrix")
plt.tight_layout()
plt.show()

results = grid.cv_results_
mean_scores = results["mean_test_score"]
neighbor_values = results["param_knn__n_neighbors"]
weight_values = results["param_knn__weights"]
metric_values = results["param_knn__metric"]

plt.figure(figsize=(8, 5))

for weight in ["uniform", "distance"]:
    for metric in ["euclidean", "manhattan"]:
        scores = [
            mean_scores[i] * 100
            for i in range(len(mean_scores))
            if weight_values[i] == weight
            and metric_values[i] == metric
        ]
        ks = [
            int(neighbor_values[i])
            for i in range(len(mean_scores))
            if weight_values[i] == weight
            and metric_values[i] == metric
        ]

        plt.plot(ks, scores, marker="o",
                 label=f"{weight}, {metric}")

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy (%)")
plt.title("kNN Hyperparameter Tuning")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


