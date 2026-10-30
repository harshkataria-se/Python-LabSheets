import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

data = {
    "Text": [
        "The students are studying different subjects",
        "He plays games and runs daily",
        "The children were playing in the gardens"
    ]
}

df = pd.DataFrame(data)

lemmatizer = WordNetLemmatizer()

df["Lemmatized Text"] = df["Text"].apply(
    lambda text: " ".join(
        lemmatizer.lemmatize(word) for word in word_tokenize(text)
    )
)

print("Original Text:")
print(df["Text"])

print("\nLemmatized Text:")
print(df["Lemmatized Text"])