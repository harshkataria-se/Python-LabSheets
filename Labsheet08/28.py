import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    "Text": [
        "I LOVE Machine Learning!!!",
        "Python is very EASY to learn.",
        "This movie is absolutely excellent!",
        "The service was really poor.",
        "AI is the FUTURE of technology.",
        "The product is very BAD.",
        "I enjoy programming a lot.",
        "The experience was terrible!!!",
        "Technology is AMAZING.",
        "The food was disappointing.",
        "Machine learning is very useful.",
        "The movie was extremely awful.",
        "Python programming is interesting!",
        "The service is excellent.",
        "The product is terrible."
    ],
    "Label": [
        "Positive", "Positive", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Negative",
        "Positive", "Negative", "Positive", "Positive", "Negative"
    ]
}

df = pd.DataFrame(data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

df["Cleaned Text"] = df["Text"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"],
    df["Label"],
    test_size=0.3,
    random_state=42,
    stratify=df["Label"]
)

X_train_clean = X_train.apply(preprocess_text)
X_test_clean = X_test.apply(preprocess_text)

vectorizer_before = TfidfVectorizer()
X_train_before = vectorizer_before.fit_transform(X_train)
X_test_before = vectorizer_before.transform(X_test)

model_before = MultinomialNB()
model_before.fit(X_train_before, y_train)
pred_before = model_before.predict(X_test_before)

accuracy_before = accuracy_score(y_test, pred_before)

vectorizer_after = TfidfVectorizer(stop_words="english")
X_train_after = vectorizer_after.fit_transform(X_train_clean)
X_test_after = vectorizer_after.transform(X_test_clean)

model_after = MultinomialNB()
model_after.fit(X_train_after, y_train)
pred_after = model_after.predict(X_test_after)

accuracy_after = accuracy_score(y_test, pred_after)

results = pd.DataFrame({
    "Model": ["Before Preprocessing", "After Preprocessing"],
    "Accuracy": [accuracy_before, accuracy_after]
})

print("Performance Comparison:")
print(results.to_string(index=False))

print("\nAccuracy Before Preprocessing:", accuracy_before)
print("Accuracy After Preprocessing:", accuracy_after)