import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("genetic_data_train.csv")

# Basic summary
print(train_df.head())
print(train_df.info())

# Sequence length distribution
train_df['sequence_length'] = train_df['sequence'].apply(len)

sns.histplot(train_df['sequence_length'], kde=True)
plt.title("DNA Sequence Length Distribution")
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.show()
