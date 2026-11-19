
import pandas as pd

# Load the downloaded dataset
df = pd.read_csv("data/publications.csv")

# Display dataset dimensions
print("Dataset shape:", df.shape)

# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Display first five publications
print("\nFirst five publications:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate publications
print("\nDuplicate paper IDs:")
print(df["paper_id"].duplicated().sum())

# Calculate abstract lengths
df["abstract_length"] = df["abstract"].str.len()

print("\nAbstract statistics:")
print(df["abstract_length"].describe())

# Examine publication categories
print("\nMost frequent categories:")
print(
    df["categories"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)
