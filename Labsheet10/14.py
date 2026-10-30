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

pca = PCA()
pca.fit(scaled_data)

cumulative_variance = pca.explained_variance_ratio_.cumsum()

minimum_components = (cumulative_variance >= 0.95).argmax() + 1

print("Cumulative Explained Variance:")
print(cumulative_variance)

print("\nMinimum Number of Components Required:")
print(minimum_components)

print("\nVariance Explained:")
print(cumulative_variance[minimum_components - 1] * 100, "%")