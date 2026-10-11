import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'Name': ['Aman', 'Riya', 'Rahul', 'Riya', 'Aman'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])

print("\nDataset After Label Encoding:")
print(df)