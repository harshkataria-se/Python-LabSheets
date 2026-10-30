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

cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker='o'
)

plt.axhline(y=0.95, linestyle='--', label='95% Variance')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.show()