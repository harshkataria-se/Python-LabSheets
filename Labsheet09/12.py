import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Message_Length": [25, 40, 15, 60, 35, 50, 20, 45, 70, 65],
    "Word_Count": [5, 8, 3, 12, 7, 10, 4, 9, 14, 13]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(scaled_data)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 8))
plt.grid(True)
plt.show()

optimal_k = 2

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_data)

print("\nDataset After Clustering:")
print(df)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Centers:")
print(centers)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Message_Length"],
    df["Word_Count"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Message Length")
plt.ylabel("Word Count")
plt.title("K-Means Clustering Using Optimal K = 2")
plt.legend()
plt.grid(True)
plt.show()