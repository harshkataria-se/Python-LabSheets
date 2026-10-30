import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Text": [
        "I love machine learning",
        "Python is easy to learn",
        "This movie is excellent",
        "The service was poor",
        "AI is the future",
        "The product is bad",
        "I enjoy programming",
        "The experience was terrible",
        "Technology is amazing",
        "The food was disappointing",
        "Machine learning is useful",
        "The movie was awful",
        "Python programming is interesting",
        "The service is excellent",
        "The product is terrible"
    ],
    "Label": [
        "Positive", "Positive", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Negative",
        "Positive", "Negative", "Positive", "Positive", "Negative"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"], df["Label"],
    test_size=0.3, random_state=42, stratify=df["Label"]
)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

count_model = MultinomialNB()
count_model.fit(X_train_count, y_train)
count_pred = count_model.predict(X_test_count)

count_accuracy = accuracy_score(y_test, count_pred)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

tfidf_model = MultinomialNB()
tfidf_model.fit(X_train_tfidf, y_train)
tfidf_pred = tfidf_model.predict(X_test_tfidf)

tfidf_accuracy = accuracy_score(y_test, tfidf_pred)

results = pd.DataFrame({
    "Vectorizer": ["CountVectorizer", "TfidfVectorizer"],
    "Accuracy": [count_accuracy, tfidf_accuracy]
})

print("Comparison of CountVectorizer and TF-IDF Vectorizer")
print(results.to_string(index=False))

print("\nClassification Report using CountVectorizer:")
print(classification_report(
    y_test, count_pred, zero_division=0
))

print("Classification Report using TF-IDF Vectorizer:")
print(classification_report(
    y_test, tfidf_pred, zero_division=0
))