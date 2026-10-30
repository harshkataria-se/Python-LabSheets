from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model1 = KNeighborsClassifier(n_neighbors=5)
model1.fit(X_train_scaled, y_train)

y_pred1 = model1.predict(X_test_scaled)
accuracy_before = accuracy_score(y_test, y_pred1)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_pca, y_train)

y_pred2 = model2.predict(X_test_pca)
accuracy_after = accuracy_score(y_test, y_pred2)

print("Classification Accuracy Before PCA:")
print(accuracy_before * 100, "%")

print("\nClassification Accuracy After PCA:")
print(accuracy_after * 100, "%")

print("\nAccuracy Comparison:")
print("Before PCA:", accuracy_before * 100, "%")
print("After PCA:", accuracy_after * 100, "%")