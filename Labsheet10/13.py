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

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nCumulative Explained Variance:")
for i, value in enumerate(cumulative_variance):
    print(f"First {i + 1} Components: {value * 100:.2f}%")

print("\nTotal Cumulative Explained Variance:")
print(cumulative_variance[-1] * 100, "%")