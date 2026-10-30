import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

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

kmeans_score = silhouette_score(scaled_data, kmeans_labels)
hierarchical_score = silhouette_score(scaled_data, hierarchical_labels)

print("Clustering Performance Comparison")
print("---------------------------------")
print("K-Means Silhouette Score:", kmeans_score)
print("Hierarchical Silhouette Score:", hierarchical_score)

if kmeans_score > hierarchical_score:
    print("\nK-Means performs better.")
else:
    print("\nHierarchical Clustering performs better.")

scores = [kmeans_score, hierarchical_score]
methods = ["K-Means", "Hierarchical"]

plt.bar(methods, scores)
plt.xlabel("Clustering Algorithms")
plt.ylabel("Silhouette Score")
plt.title("Clustering Performance Comparison")
plt.show()