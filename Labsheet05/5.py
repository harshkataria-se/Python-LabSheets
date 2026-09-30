import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'Study_Level': ['Low', 'Medium', 'High', 'Medium', 'High'],
    'Result': ['Fail', 'Fail', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Study_Level'] = encoder.fit_transform(df['Study_Level'])
df['Result'] = encoder.fit_transform(df['Result'])

print("\nDataset After Label Encoding:")
print(df)