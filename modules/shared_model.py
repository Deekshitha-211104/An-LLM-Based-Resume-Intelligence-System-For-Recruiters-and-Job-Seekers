# shared_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Single shared SentenceTransformer instance.
# Both embedding_engine.py and jd_processor.py import from here so the
# model is only loaded ONCE, saving ~500 MB of RAM in Colab.
# ─────────────────────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer

print("[shared_model] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("[shared_model] SentenceTransformer loaded.")