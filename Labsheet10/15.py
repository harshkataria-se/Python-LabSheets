import pandas as pd
import matplotlib.pyplot as plt
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

explained_variance = pca.explained_variance_ratio_

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance,
    marker='o'
)

plt.title("Scree Plot")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()