
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Load clustering results
# --------------------------------------------------

df = pd.read_csv(
    "outputs/publications_clustered.csv"
)

embeddings = np.load(
    "outputs/embeddings/sbert_embeddings.npy"
)

print("Total publications:", len(df))
print("Number of clusters:", df["cluster"].nunique())


# --------------------------------------------------
# 2. Evaluate overall clustering quality
# --------------------------------------------------

score = silhouette_score(
    embeddings,
    df["cluster"],
    metric="cosine"
)

print("\nOverall silhouette score:", round(score, 4))


# --------------------------------------------------
# 3. Examine cluster sizes
# --------------------------------------------------

print("\nCluster distribution:")

print(
    df["cluster"].value_counts().sort_index()
)


# --------------------------------------------------
# 4. Identify important terms in each cluster
# --------------------------------------------------

# TF-IDF is used only for cluster interpretation.
# It does NOT replace the SBERT embeddings.

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(
    df["text_clean"].fillna("")
)

terms = np.array(
    vectorizer.get_feature_names_out()
)


# --------------------------------------------------
# 5. Display the most representative terms
# --------------------------------------------------

print("\nIMPORTANT TERMS BY CLUSTER")

for cluster_id in sorted(df["cluster"].unique()):

    cluster_indices = np.where(
        df["cluster"].values == cluster_id
    )[0]

    # Calculate average TF-IDF score per cluster
    cluster_tfidf = tfidf_matrix[
        cluster_indices
    ].mean(axis=0)

    scores = np.asarray(
        cluster_tfidf
    ).flatten()

    top_indices = np.argsort(
        scores
    )[::-1][:15]

    print(f"\nCLUSTER {cluster_id}")

    for index in top_indices:
        print(
            f"{terms[index]}: {scores[index]:.4f}"
        )


# --------------------------------------------------
# 6. Examine category proportions
# --------------------------------------------------

print("\nCATEGORY COVERAGE BY CLUSTER")

for cluster_id in sorted(df["cluster"].unique()):

    cluster_df = df[
        df["cluster"] == cluster_id
    ]

    category_lists = (
        cluster_df["categories"]
        .dropna()
        .str.split(", ")
    )

    category_counts = (
        category_lists.explode()
        .value_counts()
    )

    # Divide by cluster size because categories
    # are multi-label and can overlap.

    category_percentages = (
        category_counts / len(cluster_df) * 100
    )

    print(f"\nCLUSTER {cluster_id}")

    print(
        category_percentages.head(10).round(2)
    )


print("\nCluster interpretation completed!")
