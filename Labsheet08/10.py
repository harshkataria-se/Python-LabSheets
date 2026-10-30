import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    "Text": [
        "I love machine learning",
        "Python is easy to learn",
        "I love Python programming"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Text"])

feature_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(
    X.toarray(),
    columns=feature_names
)

print("Original Text:")
print(df)

print("\nVocabulary:")
print(feature_names)

print("\nTF-IDF Feature Matrix:")
print(df_tfidf.round(3))