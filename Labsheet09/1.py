import pandas as pd

data = {
    "Message": [
        "Congratulations! You won a prize",
        "Please call me tomorrow",
        "Get a free gift now",
        "Are you coming to college?",
        "Claim your reward today"
    ]
}

df = pd.DataFrame(data)

print("Unlabeled Dataset:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nFirst Five Records:")
print(df.head())