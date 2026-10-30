import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92],
    "Salary": [25000, 30000, 22000, 28000, 32000]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

covariance_matrix = np.cov(scaled_data, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

print("Covariance Matrix:")
print(covariance_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)