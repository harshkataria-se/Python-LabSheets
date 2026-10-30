import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        "The food was disappointing"
    ],
    "Label": [
        "Positive", "Positive", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Negative"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["Text"], df["Label"], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

cm = confusion_matrix(
    y_test, y_pred, labels=["Negative", "Positive"]
)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()