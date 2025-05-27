import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming you have a list or pandas Series of sequences
# Example (replace with your actual sequence data)
sequences = ["ATGCGT", "CGATGC", "TTAGC", "AGCTAGCTAGCT"]

# Calculate sequence lengths
sequence_lengths = [len(seq) for seq in sequences]

# Create a pandas Series for easier plotting
sequence_length_series = pd.Series(sequence_lengths)

# Visualize the distribution
plt.figure(figsize=(10, 6))
sns.histplot(sequence_length_series, kde=True, color='skyblue')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()
