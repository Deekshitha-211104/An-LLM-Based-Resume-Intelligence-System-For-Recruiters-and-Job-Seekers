# ranking_engine.py
import faiss
import numpy as np

DIMENSION = 384

# ── Module-level mutable containers ───────────────────────────────────────────
# We use a plain list and recreate the FAISS index in-place so that
# application.py's  `resume_meta.clear()`  and  `index.reset()`  always
# operate on the SAME objects — no stale reference bugs.

resume_meta = []          # list of dicts, one per resume
_faiss_index = faiss.IndexFlatIP(DIMENSION)


class _IndexProxy:
    """
    Thin proxy so application.py can call  index.reset()
    and still have `index` point at the live FAISS object.
    """
    def reset(self):
        global _faiss_index
        _faiss_index = faiss.IndexFlatIP(DIMENSION)
        resume_meta.clear()          # clear the shared list in-place

    # Make len(index) work
    def __len__(self):
        return len(resume_meta)


index = _IndexProxy()


# ── Public helpers called by application.py ───────────────────────────────────

def add_resume_embedding(embedding, meta):
    global _faiss_index
    vec = np.array([embedding], dtype=np.float32)
    _faiss_index.add(vec)
    resume_meta.append(meta)


def search_candidates(jd_embedding, top_k=10):
    total = len(resume_meta)
    if total == 0:
        return []

    k = min(top_k, total)
    query = np.array([jd_embedding], dtype=np.float32)
    scores, indices = _faiss_index.search(query, k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue
        meta = resume_meta[idx]
        results.append({
            "rank":     rank + 1,
            "name":     meta.get("name",   "Unknown"),
            "skills":   meta.get("skills", []),
            "email":    meta.get("email",  "N/A"),
            "phone":    meta.get("phone",  "N/A"),
            "score":    round(float(score), 4),   # lowercase — app reads r.get("score")
            "metadata": meta,
        })

    return results