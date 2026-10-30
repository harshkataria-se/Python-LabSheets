import pandas as pd
from io import StringIO

text_data = """Text,Label
"I love machine learning",Positive
"This movie is boring",Negative
"Python is easy to learn",Positive
"The service was poor",Negative
"""

df = pd.read_csv(StringIO(text_data))

print("Dataset Structure:")
df.info()

print("\nDataset Summary:")
print(df.describe(include="all"))

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())