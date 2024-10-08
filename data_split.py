import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from CSV file
data = pd.read_csv('data/dataset1.csv')

# Shuffle the data randomly
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data with 80% for training and 20% for testing
train_data, test_data = train_test_split(data, test_size=0.05, random_state=42)

# Save the train and test data to separate CSV files
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)

print("Data has been split and saved as train.csv and test.csv")
