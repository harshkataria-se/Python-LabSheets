import pandas as pd

data = {
    "Name": ["Amit", "Rahul", "Priya", "Neha", "Karan"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 90, 78, 88, 92]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

df.to_csv("student_dataset.csv", index=False)

print("\nDataset saved successfully.")