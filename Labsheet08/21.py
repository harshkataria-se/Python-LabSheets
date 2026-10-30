import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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
        "The product is terrible",
        "I enjoy artificial intelligence",
        "The food is delicious",
        "The experience was bad",
        "Programming is fun",
        "The movie was boring"
    ],
    "Label": [
        "Positive", "Positive", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Negative",
        "Positive", "Negative", "Positive", "Positive", "Negative",
        "Positive", "Positive", "Negative", "Positive", "Negative"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"],
    df["Label"],
    test_size=0.3,
    random_state=42,
    stratify=df["Label"]
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Bernoulli Naive Bayes": BernoulliNB()
}

results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Classifier": name,
        "Accuracy": accuracy
    })

    print("\n", name)
    print("Actual Labels:", y_test.tolist())
    print("Predicted Labels:", y_pred.tolist())
    print(classification_report(y_test, y_pred, zero_division=0))

print("\nClassifier Comparison:")
print(pd.DataFrame(results).to_string(index=False))