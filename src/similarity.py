
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Load cleaned publications
# --------------------------------------------------

df = pd.read_csv("data/publications_cleaned.csv")


# --------------------------------------------------
# 2. Load previously generated SBERT embeddings
# --------------------------------------------------

embeddings = np.load(
    "outputs/embeddings/sbert_embeddings.npy"
)

print("Embedding matrix shape:", embeddings.shape)


# --------------------------------------------------
# 3. Compute cosine similarity
# --------------------------------------------------

# Compare every publication with every other publication.
#
# Since embeddings are normalized, cosine similarity
# is equivalent to their dot product.

similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity matrix shape:",
      similarity_matrix.shape)


# --------------------------------------------------
# 4. Select one publication
# --------------------------------------------------

# We use the first publication as our reference.

reference_index = 0

reference_title = df.iloc[
    reference_index
]["title"]

print("\nReference publication:")
print(reference_title)


# --------------------------------------------------
# 5. Find the most similar publications
# --------------------------------------------------

similarities = similarity_matrix[reference_index]

# Sort scores in descending order.
# Exclude the reference publication itself.

similar_indices = np.argsort(
    similarities
)[::-1][1:6]


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

results = []

for index in similar_indices:

    results.append({

        "Title": df.iloc[index]["title"],

        "Similarity": round(
            float(similarities[index]), 4
        ),

        "Categories": df.iloc[index]["categories"]

    })


results_df = pd.DataFrame(results)

print("\nTop 5 semantically similar publications:")

print(results_df.to_string(index=False))


# --------------------------------------------------
# 7. Save similarity results
# --------------------------------------------------

results_df.to_csv(
    "outputs/similar_publications.csv",
    index=False
)

print("\nResults saved successfully!")
