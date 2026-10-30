import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = {
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "Salary": [25000, 30000, 22000, 28000, 32000],
    "Study_Hours": [3, 4, 2, 5, 4]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

reduced_df = pd.DataFrame(
    reduced_data,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nDataset After Dimensionality Reduction:")
print(reduced_df)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nOriginal Dimensions:", df.shape[1])
print("Reduced Dimensions:", reduced_df.shape[1])