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

k_values = [2, 3, 4, 5]

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df[f"Cluster_K{k}"] = kmeans.fit_predict(scaled_data)

    print(f"\nCluster Assignments for K = {k}:")
    print(df[["Message_Length", "Word_Count", f"Cluster_K{k}"]])

    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["Message_Length"],
        df["Word_Count"],
        c=df[f"Cluster_K{k}"],
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
    plt.title(f"K-Means Clustering (K = {k})")
    plt.legend()
    plt.grid(True)
    plt.show()