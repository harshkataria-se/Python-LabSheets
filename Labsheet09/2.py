import pandas as pd

data = {
    "Message": [
        "Congratulations! You won a prize",
        "Please call me tomorrow",
        "Get a free gift now",
        "Are you coming to college?",
        "Claim your reward today"
    ],
    "Length": [34, 24, 20, 27, 24],
    "Word_Count": [5, 4, 5, 5, 4]
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