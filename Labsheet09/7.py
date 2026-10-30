import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Message_Length": [25, 40, 15, 60, 35, 50, 20, 45],
    "Word_Count": [5, 8, 3, 12, 7, 10, 4, 9]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(scaled_data)

print("\nDataset After K-Means Clustering:")
print(df)

print("\nCluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Message_Length"],
    df["Word_Count"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

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
plt.title("K-Means Clustering (K = 2)")
plt.legend()
plt.grid(True)
plt.show()