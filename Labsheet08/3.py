import pandas as pd

data = {
    "Text": [
        "I Love Machine Learning",
        "PYTHON Is Easy To Learn",
        "TEXT PREPROCESSING IS IMPORTANT"
    ]
}

df = pd.DataFrame(data)

print("Original Text:")
print(df)

df["Text"] = df["Text"].str.lower()

print("\nText After Lowercase Conversion:")
print(df)