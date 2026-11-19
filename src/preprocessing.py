
import pandas as pd
import re

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/publications.csv")

print("Original dataset shape:", df.shape)


# --------------------------------------------------
# 2. Remove missing values
# --------------------------------------------------

# Publications without titles or abstracts cannot
# contribute meaningfully to semantic analysis.

df = df.dropna(subset=["title", "abstract"])


# --------------------------------------------------
# 3. Remove duplicate publications
# --------------------------------------------------

# Paper IDs uniquely identify arXiv publications.

df = df.drop_duplicates(subset=["paper_id"])


# --------------------------------------------------
# 4. Define text cleaning function
# --------------------------------------------------

def clean_text(text):

    # Convert text to string
    text = str(text)

    # Remove unnecessary line breaks and tabs
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


# --------------------------------------------------
# 5. Clean titles and abstracts
# --------------------------------------------------

df["title_clean"] = df["title"].apply(clean_text)

df["abstract_clean"] = df["abstract"].apply(clean_text)


# --------------------------------------------------
# 6. Remove empty abstracts
# --------------------------------------------------

df = df[
    df["abstract_clean"].str.len() > 0
].copy()


# --------------------------------------------------
# 7. Combine title and abstract
# --------------------------------------------------

# The title provides additional context about
# the scientific publication.

df["text_clean"] = (
    df["title_clean"] + ". " + df["abstract_clean"]
)


# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    "data/publications_cleaned.csv",
    index=False
)

print("\nPreprocessing completed!")

print("Cleaned dataset shape:", df.shape)

print("\nExample cleaned publication:")
print(df["text_clean"].iloc[0])

