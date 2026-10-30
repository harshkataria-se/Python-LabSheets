import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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

linked = linkage(scaled_data, method="ward")

plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    labels=df.index,
    leaf_rotation=0,
    leaf_font_size=10
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()