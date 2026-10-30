import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Iris Dataset:")
print(df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_data)

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)
hierarchical_labels = hierarchical.fit_predict(scaled_data)

df["KMeans_Cluster"] = kmeans_labels
df["Hierarchical_Cluster"] = hierarchical_labels

print("\nClustering Results:")
print(df.head(10))

kmeans_score = silhouette_score(scaled_data, kmeans_labels)
hierarchical_score = silhouette_score(scaled_data, hierarchical_labels)

print("\nSilhouette Scores:")
print("K-Means:", kmeans_score)
print("Hierarchical Clustering:", hierarchical_score)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=40
)
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.subplot(1, 2, 2)
plt.scatter(
    scaled_data[:, 0],
    scaled_data[:, 1],
    c=hierarchical_labels,
    cmap="viridis",
    s=40
)
plt.title("Hierarchical Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.tight_layout()
plt.show()