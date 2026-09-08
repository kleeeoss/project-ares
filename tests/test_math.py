import numpy as np
import pytest
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def test_embedding_dimensionality(model):
    text = "Kubernetes orchestration for distributed systems."
    vector = model.encode(text)
    assert vector.shape == (384,), f"Expected 384 dimensions, received {vector.shape}"


def test_l2_normalization(model):
    text = "Invariant validation for vector database indexing."
    vector = model.encode(text, normalize_embeddings=True)
    norm = np.linalg.norm(vector)
    assert np.isclose(norm, 1.0, atol=1e-5), f"Vector norm {norm} deviates from unit length 1.0"


def test_cosine_similarity_bounds(model):
    v1 = model.encode("Machine learning models in production.", normalize_embeddings=True)
    v2 = model.encode("Deep learning pipelines with PyTorch.", normalize_embeddings=True)

    # Cosine similarity for unit vectors is simply the dot product
    similarity = float(np.dot(v1, v2))
    assert -1.0 <= similarity <= 1.0, f"Cosine similarity {similarity} breached [-1.0, 1.0] bounds"


def test_exact_match_self_similarity(model):
    text = "Deterministic vector cosine similarity."
    v = model.encode(text, normalize_embeddings=True)
    similarity = float(np.dot(v, v))
    assert np.isclose(similarity, 1.0, atol=1e-5), f"Self-similarity {similarity} must equal 1.0"