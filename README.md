# Scientific Publication Clustering Using Sentence-BERT and KMeans

## Overview

This project explores semantic analysis and clustering of scientific publications using **Sentence-BERT (SBERT) embeddings** and the **KMeans clustering algorithm**.

The objective is to group research papers based on the semantic meaning of their titles and abstracts rather than relying solely on keyword matching. Publications are collected from the arXiv API across multiple computer science research areas.

The project demonstrates how transformer-based sentence embeddings can support semantic similarity search, research-paper discovery, and topic exploration.

## Key Features

- Retrieve scientific publications using the arXiv API.
- Explore and preprocess publication titles and abstracts.
- Generate semantic embeddings using Sentence-BERT.
- Calculate cosine similarity between publications.
- Identify semantically similar research papers.
- Apply KMeans clustering to group publications.
- Evaluate cluster quality using silhouette scores.
- Visualise clusters using Principal Component Analysis (PCA).
- Interpret clusters using TF-IDF keywords and arXiv categories.
- Consider computational efficiency and environmental impact.

## Dataset

The dataset contains **1,000 scientific publications** retrieved from arXiv.

The original search focused on the following categories:

| Category | Research area |
|---|---|
| `cs.AI` | Artificial Intelligence |
| `cs.LG` | Machine Learning |
| `cs.CV` | Computer Vision |
| `cs.CL` | Computation and Language |
| `cs.RO` | Robotics |

The dataset includes:

| Column | Description |
|---|---|
| `paper_id` | Unique arXiv paper identifier |
| `title` | Publication title |
| `abstract` | Publication abstract |
| `categories` | Associated arXiv categories |

The dataset was checked for missing values and duplicate paper IDs before preprocessing.

## Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Embedding Model:** Sentence-Transformers
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualisation:** Matplotlib, Seaborn
- **Data Source:** arXiv API

## Methodology

### 1. Data Collection

Scientific publications were retrieved using the arXiv API. A dataset of 1,000 publications was collected and stored in CSV format.

### 2. Data Exploration and Preprocessing

The dataset was examined for missing values, duplicate records, category distribution, and abstract length.

Text preprocessing included whitespace normalization and preparation of publication titles and abstracts for semantic embedding generation.

### 3. Semantic Embedding Generation

The pretrained model used was:

`sentence-transformers/all-MiniLM-L6-v2`

Each publication was represented by a **384-dimensional semantic embedding**.

These embeddings capture semantic relationships between research papers, allowing publications with related meanings to be identified even when they use different terminology.

The model's maximum sequence length is 256 tokens, so longer texts may be truncated.

### 4. Semantic Similarity Analysis

Cosine similarity was calculated between publication embeddings to identify papers with similar semantic content.

The analysis also retrieves the most similar publications for a selected reference paper.

### 5. KMeans Clustering

KMeans was applied to the generated SBERT embeddings to group publications according to their semantic representations.

Different values of K, ranging from 2 to 10, were evaluated using the silhouette score with cosine distance.

The highest score among the tested values was obtained with **K = 2**.

### 6. Cluster Visualisation

Principal Component Analysis (PCA) was used to reduce the 384-dimensional embeddings to two dimensions for visualisation.

PCA was used only for visualisation. KMeans clustering was performed on the original 384-dimensional embeddings.

### 7. Cluster Interpretation

TF-IDF was used to identify important terms within each cluster. The clusters were also examined using their associated arXiv categories.

The resulting clusters showed two broad themes:

- **Cluster 0 – Robotics and Embodied AI:** robot navigation, manipulation, object grounding, language-guided interaction, and real-world tasks.
- **Cluster 1 – Vision-Language and Multimodal AI:** image-text learning, visual understanding, multimodal models, and visual reasoning.

These labels are interpretive descriptions based on the cluster contents, not predefined categories.

## Results

### Silhouette Score Evaluation

| Number of clusters (K) | Silhouette score |
|---:|---:|
| 2 | 0.1318 |
| 3 | 0.1120 |
| 4 | 0.0998 |
| 5 | 0.0668 |
| 6 | 0.0700 |
| 7 | 0.0572 |
| 8 | 0.0442 |
| 9 | 0.0466 |
| 10 | 0.0460 |

The highest silhouette score was **0.1318 for K = 2**.

The relatively low score suggests that the clusters overlap and are not strongly separated. The results should therefore be interpreted as broad semantic groupings rather than perfectly distinct research topics.

### Cluster Distribution

| Cluster | Publications | Percentage |
|---|---:|---:|
| Cluster 0 | 552 | 55.2% |
| Cluster 1 | 448 | 44.8% |
| **Total** | **1,000** | **100%** |

### Visualisations

The repository includes:

- Silhouette score comparison for different cluster counts.
- PCA visualisation of the publication clusters.

See the `outputs/figures/` directory.

## Project Structure

```text
scientific-publication-clustering/
│
├── data/
│   ├── publications.csv
│   └── publications_cleaned.csv
│
├── outputs/
│   ├── embeddings/
│   │   └── sbert_embeddings.npy
│   ├── figures/
│   │   ├── silhouette_scores.png
│   │   └── publication_clusters.png
│   ├── similar_publications.csv
│   └── publications_clustered.csv
│
├── notebooks/
│   └── sbert_clustering.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── explore_dataset.py
│   ├── preprocessing.py
│   ├── generate_embeddings.py
│   ├── similarity.py
│   ├── clustering.py
│   └── cluster_interpretation.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd scientific-publication-clustering
```

Replace the placeholder with your repository URL.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook:

```text
notebooks/sbert_clustering.ipynb
```

Run the notebook cells in order.

## Output Files

| File | Description |
|---|---|
| `sbert_embeddings.npy` | Generated SBERT embedding vectors |
| `similar_publications.csv` | Publications most similar to the selected reference paper |
| `publications_clustered.csv` | Publications with their assigned cluster labels |
| `silhouette_scores.png` | Silhouette scores for different K values |
| `publication_clusters.png` | Two-dimensional PCA cluster visualisation |

The embeddings can be regenerated by running the embedding-generation steps in the notebook.

## Eco-Responsible AI Considerations

This project uses a pretrained Sentence-BERT model instead of training a large language model from scratch.

Several choices help limit unnecessary computational overhead:

- Reusing a pretrained embedding model.
- Using a relatively lightweight model, `all-MiniLM-L6-v2`.
- Generating embeddings once and saving them for reuse.
- Avoiding repeated model inference when saved embeddings are available.
- Applying PCA for visualisation without retraining the embedding model.

These choices help make semantic analysis more computationally efficient.

## Limitations

- The dataset contains 1,000 publications and may not represent the full research landscape.
- The SBERT model truncates text beyond its maximum sequence length.
- KMeans uses Euclidean distance by default, while silhouette evaluation uses cosine distance. These are not identical distance measures.
- The relatively low silhouette score indicates overlap between clusters.
- arXiv categories are multi-label, meaning a publication can belong to multiple categories.
- Cluster labels are assigned through interpretation rather than ground-truth topic labels.

## Future Improvements

Potential extensions include:

- Experimenting with alternative sentence-transformer models.
- Comparing KMeans with hierarchical clustering and density-based clustering.
- Evaluating clustering with additional metrics.
- Exploring topic modelling techniques.
- Increasing the dataset size and including more research categories.
- Building an interactive interface for semantic publication search.

## Conclusion

This project demonstrates how Sentence-BERT embeddings can represent scientific publications semantically and how KMeans can group papers based on those representations.

The analysis identified two broad research groupings related to robotics and embodied AI, and vision-language and multimodal AI.

Although the clusters are not strongly separated, the results illustrate the potential of semantic embeddings for scientific literature exploration, similarity search, and research topic discovery.

## Acknowledgements

- [arXiv](https://arxiv.org/) for providing access to scientific publications.
- [Sentence-Transformers](https://www.sbert.net/) for the semantic embedding framework.
- [Hugging Face](https://huggingface.co/) for hosting pretrained models.
