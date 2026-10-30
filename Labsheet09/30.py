import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

print("Customer Dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

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

print("\nK-Means Silhouette Score:", kmeans_score)
print("Hierarchical Silhouette Score:", hierarchical_score)

if kmeans_score > hierarchical_score:
    print("\nK-Means gives better clustering performance.")
else:
    print("\nHierarchical Clustering gives better performance.")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=kmeans_labels,
    cmap="viridis",
    s=50
)
plt.title("Customer Segmentation using K-Means")
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
plt.title("Customer Segmentation using Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.tight_layout()
plt.show()

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 5))
plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=kmeans_labels,
    cmap="viridis",
    s=50
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)
plt.title("K-Means Customer Segments with Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

df.to_csv("Customer_Segmentation_Results.csv", index=False)

print("\nCustomer Segmentation Results:")
print(df.head())

print("\nResults saved to Customer_Segmentation_Results.csv")