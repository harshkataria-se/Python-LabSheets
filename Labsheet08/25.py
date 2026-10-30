import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    "Message": [
        "Win a free prize now",
        "Congratulations you won a lottery",
        "Claim your free cash reward",
        "Get a free mobile phone today",
        "You have won a free gift",
        "Hello how are you",
        "Are you coming to college today",
        "Please call me when you are free",
        "Let's meet tomorrow",
        "I will send you the notes"
    ],
    "Label": [
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Ham", "Ham", "Ham", "Ham", "Ham"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Message"])
y = df["Label"]

model = MultinomialNB()
model.fit(X, y)

feature_names = vectorizer.get_feature_names_out()
spam_index = list(model.classes_).index("Spam")

spam_probabilities = model.feature_log_prob_[spam_index]
ham_index = list(model.classes_).index("Ham")
ham_probabilities = model.feature_log_prob_[ham_index]

word_scores = spam_probabilities - ham_probabilities

results = pd.DataFrame({
    "Word": feature_names,
    "Spam Importance Score": word_scores
})

results = results.sort_values(
    by="Spam Importance Score",
    ascending=False
)

print("Top Important Words for Spam Classification:")
print(results.head(10).to_string(index=False))