
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Load publications and SBERT embeddings
# --------------------------------------------------

df = pd.read_csv("data/publications_cleaned.csv")

embeddings = np.load(
    "outputs/embeddings/sbert_embeddings.npy"
)

print("Publications:", len(df))
print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 2. Find an appropriate number of clusters
# --------------------------------------------------

# We test different values of K.
# Silhouette score measures how well separated
# the resulting clusters are.

k_values = range(2, 11)

silhouette_scores = []

print("\nTesting different numbers of clusters...")

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    score = silhouette_score(
        embeddings,
        labels,
        metric="cosine"
    )

    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")


# --------------------------------------------------
# 3. Visualize silhouette scores
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")

plt.title("Selecting K using Silhouette Score")

plt.xticks(list(k_values))
plt.grid(alpha=0.3)

plt.tight_layout()

Path("outputs/figures").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    "outputs/figures/silhouette_scores.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Select the best K
# --------------------------------------------------

best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

print("\nSelected number of clusters:", best_k)


# --------------------------------------------------
# 5. Train final KMeans model
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(embeddings)

print("\nClustering completed!")


# --------------------------------------------------
# 6. Analyze cluster sizes
# --------------------------------------------------

cluster_counts = df["cluster"].value_counts().sort_index()

print("\nNumber of publications per cluster:")

print(cluster_counts)


# --------------------------------------------------
# 7. Reduce embeddings to 2 dimensions using PCA
# --------------------------------------------------

# PCA is used here for visualization.
# Clustering was performed on the original
# 384-dimensional SBERT embeddings.

pca = PCA(n_components=2)

embeddings_2d = pca.fit_transform(embeddings)

df["PCA1"] = embeddings_2d[:, 0]
df["PCA2"] = embeddings_2d[:, 1]


# --------------------------------------------------
# 8. Plot the clusters
# --------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["cluster"],
    cmap="tab10",
    alpha=0.7,
    s=25
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    f"Scientific Publication Clustering (K={best_k})"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.savefig(
    "outputs/figures/publication_clusters.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. Find representative publications per cluster
# --------------------------------------------------

# We identify the publications closest to each
# KMeans centroid in the original embedding space.

print("\nRepresentative publications:")

for cluster_id in range(best_k):

    cluster_indices = np.where(
        df["cluster"].values == cluster_id
    )[0]

    cluster_embeddings = embeddings[cluster_indices]

    centroid = kmeans.cluster_centers_[cluster_id]

    distances = np.linalg.norm(
        cluster_embeddings - centroid,
        axis=1
    )

    representative_positions = np.argsort(
        distances
    )[:3]

    print(f"\nCLUSTER {cluster_id}")

    for position in representative_positions:

        original_index = cluster_indices[position]

        print(
            "-",
            df.iloc[original_index]["title"]
        )


# --------------------------------------------------
# 10. Cross-check clusters with arXiv categories
# --------------------------------------------------

# Categories were NOT used to train KMeans.
# They are used only to help interpret clusters.

print("\nCategory distribution by cluster:")

for cluster_id in range(best_k):

    print(f"\nCLUSTER {cluster_id}")

    category_counts = (
        df[df["cluster"] == cluster_id]["categories"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(5)
    )

    print(category_counts)


# --------------------------------------------------
# 11. Save clustering results
# --------------------------------------------------

df.to_csv(
    "outputs/publications_clustered.csv",
    index=False
)

print("\nClustering results saved successfully!")
