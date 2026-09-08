import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
import os
from dotenv import load_dotenv

load_dotenv()

HF_token = os.getenv("HF_TOKEN")

# Suppress PyTorch warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

print("Booting Ares Vector Engine...")
print("Downloading Embedding Model (all-MiniLM-L6-v2)...")

# 1. Load the Model
# This lightweight model converts any text into a 384-dimensional vector coordinate.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Define the Knowledge
sentence_a = "The quick brown fox jumps over the lazy dog."
sentence_b = "A fast, dark-colored canine leaps above a sleepy hound."
sentence_c = "Cloud computing allows scalable infrastructure deployment."

# 3. Vectorization (Encoding)
print(" ⚙ Vectorizing text into high-dimensional space...")
vector_a = model.encode(sentence_a)
vector_b = model.encode(sentence_b)
vector_c = model.encode(sentence_c)

print(f"Dimensionality: {vector_a.shape[0]} dimensions per sentence.")

# 4. The Mathematics of Memory (Cosine Similarity)
def cosine_similarity(v1, v2):
    # Calculate the Dot Product (alignment)
    dot_product = np.dot(v1, v2)
    # Calculate the Magnitude (length of the vector)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Return the Cosine angle
    return dot_product / (norm_v1 * norm_v2)

# 5. Execute the Comparisons
sim_a_b = cosine_similarity(vector_a, vector_b)
sim_a_c = cosine_similarity(vector_a, vector_c)

print("\n--- Ares Semantic Analysis ---")
print(f"Text A: '{sentence_a}'")
print(f"Text B: '{sentence_b}'")
print(f"Text C: '{sentence_c}'")
print("---------------------------------")
print(f"Similarity (A vs B): {sim_a_b:.4f}  <-- High similarity! (Same meaning, different words)")
print(f"Similarity (A vs C): {sim_a_c:.4f}  <-- Low similarity! (Completely unrelated)")