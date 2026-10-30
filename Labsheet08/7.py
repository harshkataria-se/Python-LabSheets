import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

data = {
    "Text": [
        "The students are studying different subjects",
        "He plays games and runs daily",
        "Machine learning models are improving"
    ]
}

df = pd.DataFrame(data)

stemmer = PorterStemmer()

df["Stemmed Text"] = df["Text"].apply(
    lambda text: " ".join(
        stemmer.stem(word) for word in word_tokenize(text)
    )
)

print("Original Text:")
print(df["Text"])

print("\nStemmed Text:")
print(df["Stemmed Text"])