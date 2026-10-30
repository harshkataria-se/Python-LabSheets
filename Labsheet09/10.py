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
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

print("\nWCSS Values:")
for k, value in enumerate(wcss, start=1):
    print(f"K = {k}: {value:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, 8), wcss, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.xticks(range(1, 8))
plt.grid(True)
plt.show()