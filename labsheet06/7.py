import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    'Age': [18, 20, 22, 24, 26],
    'Salary': [20000, 25000, 30000, 35000, 40000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

print("\nStandardized Dataset:")
print(scaled_df)