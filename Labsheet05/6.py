import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'Study_Level': ['Low', 'Medium', 'High', 'Medium', 'High'],
    'Result': ['Fail', 'Fail', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

encoder = OneHotEncoder(sparse_output=False)

encoded_data = encoder.fit_transform(df)

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(df.columns)
)

print("\nDataset After One-Hot Encoding:")
print(encoded_df)