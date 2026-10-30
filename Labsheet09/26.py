import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

print("Original Dataset:")
print(df.head())

features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

model = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

df["Cluster"] = model.fit_predict(scaled_data)

print("\nDataset with Cluster Labels:")
print(df.head())

plt.figure(figsize=(8, 5))

plt.scatter(
    features["Annual Income (k$)"],
    features["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Clustering on Mall Customers Dataset")
plt.colorbar(label="Cluster Label")
plt.show()