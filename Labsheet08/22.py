import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["Label", "Message"],
    encoding="utf-8"
)

df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

X = df["Message"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

message = input("\nEnter a message to check for spam: ")
message_vec = vectorizer.transform([message])
prediction = model.predict(message_vec)[0]

if prediction == 1:
    print("Result: Spam Message")
else:
    print("Result: Not Spam (Ham)")