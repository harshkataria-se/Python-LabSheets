import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "Salary": [25000, 30000, 22000, 28000, 32000],
    "Study_Hours": [3, 4, 2, 5, 4],
    "Attendance": [85, 90, 80, 95, 88]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=3)
pca.fit(scaled_data)

print("Explained Variance Ratio of Each Component:")

for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i + 1}: {ratio:.4f}")

print("\nExplained Variance Ratio in Percentage:")
print(pca.explained_variance_ratio_ * 100)

print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum() * 100, "%")