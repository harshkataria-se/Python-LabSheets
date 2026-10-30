import pandas as pd
import string

data = {
    "Text": [
        "Hello, World!",
        "Python is easy to learn.",
        "Machine learning: AI, data, and models!"
    ]
}

df = pd.DataFrame(data)

print("Original Text:")
print(df)

df["Text"] = df["Text"].apply(
    lambda text: text.translate(str.maketrans("", "", string.punctuation))
)

print("\nText After Removing Punctuation:")
print(df)