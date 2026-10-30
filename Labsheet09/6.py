import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Message_Length": [25, 40, 15, 60, 35, 50, 20, 45],
    "Word_Count": [5, 8, 3, 12, 7, 10, 4, 9]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Message_Length"],
    df["Word_Count"],
    color="blue",
    marker="o"
)

plt.xlabel("Message Length")
plt.ylabel("Word Count")
plt.title("Scatter Plot of Dataset")
plt.grid(True)

plt.show()