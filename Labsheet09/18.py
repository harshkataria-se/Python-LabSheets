import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
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

model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

df["Cluster"] = model.fit_predict(scaled_data)

print("\nDataset with Ward Linkage Clusters:")
print(df)

plt.scatter(
    df["Message_Length"],
    df["Word_Count"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Message Length")
plt.ylabel("Word Count")
plt.title("Hierarchical Clustering Using Ward Linkage")
plt.show()