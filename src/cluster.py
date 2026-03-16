from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS


def parse_suricata_timestamp(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    if len(ts) >= 5 and (ts[-5] in ["+", "-"]) and ts[-4:].isdigit():
        ts = ts[:-2] + ":" + ts[-2:]
    return datetime.fromisoformat(ts)


def assign_candidate_group(df: pd.DataFrame, window_minutes: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["ts_dt"] = df["timestamp"].apply(parse_suricata_timestamp)

    window_seconds = window_minutes * 60
    df["window_start"] = df["ts_dt"].apply(
        lambda d: int(d.replace(tzinfo=timezone.utc).timestamp()) // window_seconds
    )

    # group by src_ip + time window
    df["candidate_group"] = df["src_ip"].astype(str) + ":" + df["window_start"].astype(str)
    return df


def _apply_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return model.fit_predict(X)


def _apply_hierarchical(X: np.ndarray, distance_threshold: float) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="average",
        metric="cosine",
    )
    return model.fit_predict(X)


def _apply_optics(X: np.ndarray, min_samples: int, xi: float, min_cluster_size: int) -> np.ndarray:
    model = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric="cosine",
        cluster_method="xi",
    )
    return model.fit_predict(X)


def _postprocess_min_cluster_size(labels: np.ndarray, min_samples: int) -> np.ndarray:
    labels = labels.astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    size_map = {int(u): int(c) for u, c in zip(unique, counts)}
    out = labels.copy()
    for i, lab in enumerate(labels):
        if size_map.get(int(lab), 0) < min_samples:
            out[i] = -1
    return out


def cluster_within_groups(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    algo: str = "dbscan",
    eps: float = 0.25,
    min_samples: int = 3,
    distance_threshold: float = 0.25,
    xi: float = 0.05,
    min_cluster_size: int = 3,
) -> pd.DataFrame:
    algo = algo.lower().strip()
    if algo not in ("dbscan", "hierarchical", "optics"):
        raise ValueError("algo must be 'dbscan', 'hierarchical', or 'optics'")

    df = df.copy()
    df["cluster_id"] = -1

    cluster_offset = 0
    for _, idxs in tqdm(df.groupby("candidate_group").indices.items(), desc=f"Clustering groups ({algo})"):
        idx_list = list(idxs)
        if len(idx_list) < min_samples:
            continue

        X = embeddings[idx_list]

        if algo == "dbscan":
            labels = _apply_dbscan(X, eps=eps, min_samples=min_samples)
        elif algo == "optics":
            labels = _apply_optics(
                X,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
            )
        else:
            labels = _apply_hierarchical(X, distance_threshold=distance_threshold)
            labels = _postprocess_min_cluster_size(labels, min_samples=min_samples)

        mapped = []
        for lab in labels:
            mapped.append(-1 if lab == -1 else cluster_offset + int(lab))
        df.loc[idx_list, "cluster_id"] = mapped

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_offset += max(n_clusters, 0)

    return df