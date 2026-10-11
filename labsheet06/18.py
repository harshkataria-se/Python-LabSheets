
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print("Comparison of Accuracy for Different k Values:")
for k, accuracy in zip(k_values, accuracies):
    print(f"k = {k}, Accuracy = {accuracy * 100:.2f}%")

best_index = accuracies.index(max(accuracies))
print("\nBest k value:", k_values[best_index])
print(f"Best Accuracy: {accuracies[best_index] * 100:.2f}%")

