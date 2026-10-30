# pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

y = np.where(y == 0, 1, 0)

imputer = SimpleImputer(strategy="median")
X[X.columns] = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0)
    })

    print("\n", name)
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )
    display.plot(cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy"
    )
    print(name, "5-Fold CV Scores:", scores)
    print(name, "Mean CV Accuracy:", scores.mean())

results_df = pd.DataFrame(results).set_index("Model")

print("\nPerformance Comparison:")
print(results_df.round(4))

results_df.plot(kind="bar", figsize=(10, 6))
plt.title("Decision Tree vs Random Forest Performance")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

feature_indices = [data.feature_names.tolist().index("mean radius"),
                   data.feature_names.tolist().index("mean texture")]

X_2d = X.iloc[:, feature_indices]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

visual_models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}

for name, model in visual_models.items():
    model.fit(X_train_2d, y_train_2d)

    DecisionBoundaryDisplay.from_estimator(
        model,
        X_2d,
        response_method="predict",
        cmap="coolwarm",
        alpha=0.35,
        xlabel="Mean Radius",
        ylabel="Mean Texture"
    )

    plt.scatter(
        X_2d.iloc[:, 0],
        X_2d.iloc[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="black",
        s=20
    )

    plt.title(name + " Decision Boundary")
    plt.show()