import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_before = LogisticRegression(max_iter=1000)

start = time.perf_counter()
model_before.fit(X_train_scaled, y_train)
y_pred_before = model_before.predict(X_test_scaled)
time_before = time.perf_counter() - start

accuracy_before = accuracy_score(y_test, y_pred_before)

pca = PCA(n_components=5)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_after = LogisticRegression(max_iter=1000)

start = time.perf_counter()
model_after.fit(X_train_pca, y_train)
y_pred_after = model_after.predict(X_test_pca)
time_after = time.perf_counter() - start

accuracy_after = accuracy_score(y_test, y_pred_after)

print("Original Dataset Features:", X.shape[1])
print("PCA Reduced Features:", X_train_pca.shape[1])

print("\nModel Performance Before PCA")
print("Accuracy:", accuracy_before * 100, "%")
print("Execution Time:", time_before, "seconds")

print("\nModel Performance After PCA")
print("Accuracy:", accuracy_after * 100, "%")
print("Execution Time:", time_after, "seconds")

print("\nClassification Report After PCA")
print(classification_report(
    y_test, y_pred_after, target_names=wine.target_names
))

results = pd.DataFrame({
    "Method": ["Before PCA", "After PCA"],
    "Accuracy": [accuracy_before * 100, accuracy_after * 100],
    "Execution Time": [time_before, time_after]
})

print("\nPerformance Comparison")
print(results)

plt.figure(figsize=(8, 5))
plt.bar(results["Method"], results["Accuracy"])
plt.xlabel("Method")
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy Before and After PCA")
plt.ylim(0, 105)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, 6),
    pca.explained_variance_ratio_.cumsum(),
    marker="o"
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()