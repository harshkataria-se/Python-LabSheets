import pandas as pd

data = {
    "Name": ["Amit", "Rahul", "Priya", "Neha", "Karan"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nDataset Structure:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nColumn Names:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)