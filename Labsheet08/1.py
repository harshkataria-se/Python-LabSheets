import pandas as pd
from io import StringIO

text_data = """Text,Label
"I love machine learning",Positive
"This movie is boring",Negative
"Python is easy to learn",Positive
"The service was poor",Negative
"""

df = pd.read_csv(StringIO(text_data))

print("Text Dataset:")
print(df)

print("\nDataset Shape:", df.shape)
print("\nDataset Columns:", df.columns.tolist())