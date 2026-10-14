import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np
import shutil

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _project_cache_dir() -> str:
    # <project_root>/hf_cache
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)

def embed_alerts(df, model_name=DEFAULT_MODEL_NAME, batch_size=128):
    cache_dir = Path(_project_cache_dir())
    embeddings_dir = cache_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    texts = df["alert_text"].tolist()
    keys = _hash_texts(texts)

    embeddings = []
    to_embed = []
    idx_map = []

    for i, k in enumerate(keys):
        f = embeddings_dir / f"{k}.npy"
        if f.exists():
            embeddings.append(np.load(f))
        else:
            embeddings.append(None)
            to_embed.append(texts[i])
            idx_map.append(i)

    if to_embed:
        try:
            model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        except Exception:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(model_name, cache_folder=str(cache_dir))

        new_embs = model.encode(
            to_embed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        for i, emb in zip(idx_map, new_embs):
            f = embeddings_dir / f"{keys[i]}.npy"
            np.save(f, emb)
            embeddings[i] = emb

    return np.vstack(embeddings).astype("float32")


def _hash_texts(texts):
    return [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]