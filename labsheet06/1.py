import pandas as pd

data = {
    "SepalLength": [5.1, 4.9, 7.0, 6.4, 5.5, 6.5],
    "SepalWidth":  [3.5, 3.0, 3.2, 3.2, 2.3, 3.0],
    "PetalLength": [1.4, 1.4, 4.7, 4.5, 4.0, 5.2],
    "PetalWidth":  [0.2, 0.2, 1.4, 1.5, 1.3, 2.0],
    "Species": [
        "Setosa", "Setosa", "Versicolor",
        "Versicolor", "Versicolor", "Virginica"
    ]
}

df = pd.DataFrame(data)
df.to_csv("classification_data.csv", index=False)


dataset = pd.read_csv("classification_data.csv")

print("Classification Dataset:")
print(dataset)

print("\nDataset Shape:", dataset.shape)