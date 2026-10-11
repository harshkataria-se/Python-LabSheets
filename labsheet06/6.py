import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    'Hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'Scores': [20, 30, 40, 50, 60, 70, 80, 90]
}

df = pd.DataFrame(data)

X = df[['Hours']]
Y = df['Scores']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
)

print("Training Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

print("\nTraining Target:")
print(Y_train)

print("\nTesting Target:")
print(Y_test)