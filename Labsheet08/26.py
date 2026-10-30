import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

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
        "Artificial intelligence is exciting",
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

X = df["Text"]
y = df["Label"]

model = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    MultinomialNB()
)

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("5-Fold Cross-Validation Results:")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i} Accuracy: {score:.2f}")

print("\nAverage Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
print("Average Accuracy Percentage:", scores.mean() * 100, "%")