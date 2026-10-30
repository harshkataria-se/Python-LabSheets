import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_accuracy * 100, "%")
print("Random Forest Accuracy:", rf_accuracy * 100, "%")

models = ["Decision Tree", "Random Forest"]
accuracies = [dt_accuracy * 100, rf_accuracy * 100]

plt.bar(models, accuracies)
plt.xlabel("Classifiers")
plt.ylabel("Accuracy (%)")
plt.title("Performance Comparison of Classifiers")
plt.ylim(0, 105)
plt.show()