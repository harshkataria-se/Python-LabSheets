import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import DecisionBoundaryDisplay

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print("\n", name)
    print("Accuracy:", accuracy * 100, "%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )
    display.plot(cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.show()

X_visual = X.iloc[:, [2, 3]]
X_train_visual, X_test_visual, y_train_visual, y_test_visual = train_test_split(
    X_visual, y, test_size=0.2, random_state=42, stratify=y
)

for name, model in models.items():
    model.fit(X_train_visual, y_train_visual)

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_visual.to_numpy(),
        response_method="predict",
        cmap="viridis",
        alpha=0.4,
        xlabel=iris.feature_names[2],
        ylabel=iris.feature_names[3]
    )

    plt.scatter(
        X_visual.iloc[:, 0],
        X_visual.iloc[:, 1],
        c=y,
        edgecolors="black",
        cmap="viridis"
    )

    plt.title(name + " Decision Boundary")
    plt.show()

plt.bar(results.keys(), [value * 100 for value in results.values()])
plt.xlabel("Classifier")
plt.ylabel("Accuracy (%)")
plt.title("Decision Tree vs Random Forest")
plt.ylim(0, 105)
plt.show()

print("\nModel Comparison:")
for name, accuracy in results.items():
    print(name, "Accuracy:", accuracy * 100, "%")