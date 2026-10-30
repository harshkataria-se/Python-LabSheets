import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "Message_Length": [25, 40, 15, 60, 35, 50, 20, 45, 70, 65],
    "Word_Count": [5, 8, 3, 12, 7, 10, 4, 9, 14, 13]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_data)

centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    df["Message_Length"],
    df["Word_Count"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Message Length")
plt.ylabel("Word Count")
plt.title("K-Means Clustering with Centroids")
plt.legend()
plt.show()