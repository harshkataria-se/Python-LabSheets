
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn1 = KNeighborsClassifier(n_neighbors=5)
knn1.fit(X_train, y_train)
y_pred1 = knn1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)
y_pred2 = knn2.predict(X_test_scaled)
accuracy2 = accuracy_score(y_test, y_pred2)

print("Accuracy Before Scaling:", accuracy1 * 100, "%")
print("Accuracy After Scaling:", accuracy2 * 100, "%")

