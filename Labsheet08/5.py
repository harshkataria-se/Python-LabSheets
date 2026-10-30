import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data = {
    "Text": [
        "I love machine learning and artificial intelligence",
        "Python is easy to learn and use",
        "This is a program for text preprocessing"
    ]
}

df = pd.DataFrame(data)

print("Original Text:")
print(df)

stop_words = ENGLISH_STOP_WORDS

df["Text"] = df["Text"].apply(
    lambda text: " ".join(
        word for word in text.lower().split()
        if word not in stop_words
    )
)

print("\nText After Removing Stop Words:")
print(df)