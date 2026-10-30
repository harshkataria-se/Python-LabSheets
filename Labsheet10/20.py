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

print("Principal Component Vectors:")

for i, component in enumerate(pca.components_):
    print(f"\nPrincipal Component {i + 1}:")
    print(component)

print("\nAll Principal Component Vectors:")
print(pca.components_)