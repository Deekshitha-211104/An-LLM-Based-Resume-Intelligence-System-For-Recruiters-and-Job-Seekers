# embedding_engine.py
import numpy as np
from modules.shared_model import sentence_model


def resume_to_text(parsed_resume):
    """Convert parsed resume dict to a plain text string for embedding."""
    if not isinstance(parsed_resume, dict):
        return str(parsed_resume)

    parts = []

    name = parsed_resume.get("name", "")
    if name and name != "Unknown":
        parts.append(name)

    skills = parsed_resume.get("skills", [])
    if skills:
        parts.append("Skills: " + ", ".join(skills))

    for field in ("education", "experience", "projects"):
        val = parsed_resume.get(field, "")
        if val:
            parts.append(val)

    return " ".join(parts).strip()


def generate_embedding(parsed_resume):
    """
    Generate a normalized float32 embedding for a parsed resume.
    normalize_embeddings=True is required so that FAISS IndexFlatIP
    computes cosine similarity correctly.
    """
    text = resume_to_text(parsed_resume)
    if not text:
        text = "empty resume"

    embedding = sentence_model.encode(text, normalize_embeddings=True)
    return embedding.astype(np.float32)