import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Article": [
        "The government announced new economic policies",
        "Stock market prices increased today",
        "The company reported higher profits",
        "The central bank changed interest rates",
        "The cricket team won the final match",
        "Football players performed brilliantly",
        "The tournament will begin next week",
        "The player scored a winning goal",
        "New artificial intelligence technology was launched",
        "Scientists developed a new computer processor",
        "Software companies introduced new applications",
        "Technology is changing the digital world"
    ],
    "Category": [
        "Business", "Business", "Business", "Business",
        "Sports", "Sports", "Sports", "Sports",
        "Technology", "Technology", "Technology", "Technology"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["Article"],
    df["Category"],
    test_size=0.25,
    random_state=42,
    stratify=df["Category"]
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy Score:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

article = input("\nEnter a news article: ")

article_vec = vectorizer.transform([article])
prediction = model.predict(article_vec)

print("\nPredicted News Category:", prediction[0])