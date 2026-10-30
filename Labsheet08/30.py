import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

data = {
    "Message": [
        "Congratulations! You have won a free prize",
        "Claim your free cash reward now",
        "Win a brand new mobile phone today",
        "You have won a lottery worth 50000",
        "Click here to claim your reward",
        "Free entry in a weekly competition",
        "Get a free gift card now",
        "Urgent! Claim your cash prize",
        "You won a free vacation package",
        "Congratulations, you are a lucky winner",
        "Hey, are you coming to college today?",
        "Please call me when you are free",
        "Can we meet tomorrow afternoon?",
        "Your assignment is due next Monday",
        "I will reach home by evening",
        "Don't forget to bring your notebook",
        "Are you attending the class today?",
        "Let's have lunch together",
        "Please send me the project details",
        "I will call you after the meeting"
    ],
    "Category": [
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Ham", "Ham", "Ham", "Ham", "Ham",
        "Ham", "Ham", "Ham", "Ham", "Ham"
    ]
}

df = pd.DataFrame(data)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Message"] = df["Message"].apply(preprocess_text)

X = df["Message"]
y = df["Category"]

vectorizer = TfidfVectorizer(stop_words="english")
X_features = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Spam", zero_division=0)
recall = recall_score(y_test, y_pred, pos_label="Spam", zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label="Spam", zero_division=0)

print("Spam Detection System Evaluation")
print("--------------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=["Ham", "Spam"])

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Spam Detection Confusion Matrix")
plt.show()

cv_scores = cross_val_score(
    MultinomialNB(),
    X_features,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Accuracy:", cv_scores.mean())

print("\nTest Your Own Message")
message = input("Enter a message: ")

processed_message = preprocess_text(message)
message_features = vectorizer.transform([processed_message])

prediction = model.predict(message_features)

print("Predicted Category:", prediction[0])