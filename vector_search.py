import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
from dotenv import load_dotenv
import os

# Clean console output
warnings.filterwarnings("ignore")

load_dotenv()
HF_token = os.getenv("HF_TOKEN")

print("Booting Ares Search Index...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. The "Database" (Corpus)
corpus = [
    "A fast, dark-colored canine leaps above a sleepy hound.",
    "Cloud computing allows scalable infrastructure deployment.",
    "The Apollo 11 mission landed humans on the moon in 1969.",
    "Dogs are highly social animals and love to play fetch.",
    "Machine learning models require large amounts of training data."
]

query = "Tell me about space exploration and NASA."

# 2. Embed the Database into a Matrix (5 rows x 384 columns)
print(" ⚙ Indexing Corpus into Vector Space...")
corpus_embeddings = model.encode(corpus)

# 3. Embed the User Query (1 row x 384 columns)
print(" ⚙ Encoding Query...")
query_embedding = model.encode(query)

# 4. ⚡ MASSIVE PARALLEL MATH (Vectorized Cosine Similarity)
# We use np.dot to multiply the 1x384 query against the 5x384 corpus matrix all at once!
dot_products = np.dot(corpus_embeddings, query_embedding)

# Calculate magnitudes for the denominator
query_norm = np.linalg.norm(query_embedding)
corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)

# Divide to get the final cosine similarities array
similarities = dot_products / (query_norm * corpus_norms)

# 5. Top-K Sorting (Find the highest scores)
# argsort sorts from lowest to highest, so we slice with [::-1] to reverse it
top_k_indices = np.argsort(similarities)[::-1]

print("\n=========================================")
print(f"USER QUERY: '{query}'")
print("=========================================\n")

# Print the Top 3 nearest neighbors
for i in range(3):
    idx = top_k_indices[i]
    print(f"Rank {i+1} (Score: {similarities[idx]:.4f})")
    print(f"Document: {corpus[idx]}\n")