import pandas as pd
from sklearn.model_selection import train_test_split

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

X = df["Text"]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Text:")
print(X_train)

print("\nTesting Text:")
print(X_test)

print("\nTraining Labels:")
print(y_train)

print("\nTesting Labels:")
print(y_test)

print("\nTraining Set Size:", len(X_train))
print("Testing Set Size:", len(X_test))