
import pandas as pd
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_FILE = "data/publications_cleaned.csv"

OUTPUT_DIR = Path("outputs/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = OUTPUT_DIR / "sbert_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# --------------------------------------------------
# 2. Select computational device
# --------------------------------------------------

# Use Apple's Metal Performance Shaders when available.
# Otherwise, use CUDA if available, or the CPU.

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Selected device:", device)


# --------------------------------------------------
# 3. Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

# Combine titles and abstracts into one text field.
texts = df["text_clean"].fillna("").tolist()

print("Number of publications:", len(texts))


# --------------------------------------------------
# 4. Load pretrained SBERT model
# --------------------------------------------------

print("\nLoading SBERT model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

print("SBERT model loaded successfully!")

print("Model maximum sequence length:",
      model.max_seq_length)


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

# Each publication is transformed into a
# 384-dimensional numerical vector.
#
# Batch processing reduces memory usage.
#
# Normalization makes cosine similarity calculation
# easier and more efficient.

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)


# --------------------------------------------------
# 6. Save embeddings
# --------------------------------------------------

np.save(EMBEDDINGS_FILE, embeddings)

print("\nEmbeddings successfully generated!")

print("Embedding shape:", embeddings.shape)

print("Embedding dimensions:", embeddings.shape[1])

print("Saved at:", EMBEDDINGS_FILE)


# --------------------------------------------------
# 7. Inspect one embedding
# --------------------------------------------------

print("\nFirst publication embedding:")

print(embeddings[0])

print("\nFirst five numerical values:")

print(embeddings[0][:5])

