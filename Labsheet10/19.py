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

print("Original Dataset:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1", "PC2", "PC3"]
)

print("\nPCA Transformed Dataset:")
print(pca_df)

print("\nComparison of Dataset Dimensions:")
print("Original Dataset Shape:", df.shape)
print("Transformed Dataset Shape:", pca_df.shape)

print("\nTotal Explained Variance:")
print(pca.explained_variance_ratio_.sum() * 100, "%")