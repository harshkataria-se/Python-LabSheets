from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

print("Accuracy Scores for 5 Folds:")
print(scores)

print("Mean Accuracy:", scores.mean())
print("Mean Accuracy Percentage:", scores.mean() * 100, "%")