import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

print("Original Dataset:")
print(df.head())

features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster_Label"] = kmeans.fit_predict(scaled_data)

df.to_csv("Mall_Customers_Clusters.csv", index=False)

print("\nDataset with Cluster Labels:")
print(df.head())

print("\nCluster labels successfully saved to Mall_Customers_Clusters.csv")