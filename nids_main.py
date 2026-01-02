import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# Print column names
print("Columns in the dataset:")
print(df.columns)

print("\nFirst 5 rows of the dataset:")
print(df.head())
