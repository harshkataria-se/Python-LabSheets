import pandas as pd
from nltk.tokenize import word_tokenize

data = {
    "Text": [
        "I love machine learning.",
        "Python is easy to learn!",
        "Text processing is important."
    ]
}

df = pd.DataFrame(data)

print("Original Text:")
print(df)

df["Tokens"] = df["Text"].apply(word_tokenize)

print("\nTokenized Text:")
print(df[["Text", "Tokens"]])