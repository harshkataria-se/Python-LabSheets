import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print("Original Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()

features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

hierarchical = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)
hierarchical_labels = hierarchical.fit_predict(scaled_data)

df["KMeans_Cluster"] = kmeans_labels
df["Hierarchical_Cluster"] = hierarchical_labels

kmeans_score = silhouette_score(scaled_data, kmeans_labels)
hierarchical_score = silhouette_score(scaled_data, hierarchical_labels)

print("\nClustering Evaluation:")
print("K-Means Silhouette Score:", kmeans_score)
print("Hierarchical Silhouette Score:", hierarchical_score)

print("\nDataset with Cluster Labels:")
print(df.head())

df.to_csv("Complete_Clustering_Results.csv", index=False)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=kmeans_labels,
    cmap="viridis",
    s=50
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.subplot(1, 2, 2)
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=hierarchical_labels,
    cmap="viridis",
    s=50
)
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.tight_layout()
plt.show()

print("\nResults saved to Complete_Clustering_Results.csv")