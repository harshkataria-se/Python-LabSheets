import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "Text": [
        "I love machine learning",
        "Python is very easy to learn",
        "This movie is excellent",
        "The service was very poor",
        "Artificial intelligence is amazing",
        "The product is really bad",
        "I enjoy programming",
        "The experience was terrible",
        "Technology is wonderful",
        "The food was disappointing",
        "Machine learning is very useful",
        "The movie was awful",
        "Python programming is interesting",
        "The service is excellent",
        "The product is terrible",
        "AI technology is fantastic",
        "The food is delicious",
        "The experience was bad",
        "Programming is enjoyable",
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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

df["Cleaned Text"] = df["Text"].apply(preprocess_text)

X = df["Cleaned Text"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed Successfully")

print("\nActual Labels:")
print(y_test.tolist())

print("\nPredicted Labels:")
print(y_pred.tolist())

print("\nAccuracy Score:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Negative", "Positive"]
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Naive Bayes Text Classification")
plt.show()

message = input("\nEnter a text message for classification: ")

cleaned_message = preprocess_text(message)
message_vec = vectorizer.transform([cleaned_message])

prediction = model.predict(message_vec)

print("\nPredicted Class:", prediction[0])