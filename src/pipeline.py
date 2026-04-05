
import argparse
import json
import os
from typing import Optional

import numpy as np

from .parse_suricata import parse_suricata_alerts
from .embed import embed_alerts, DEFAULT_MODEL_NAME
from .cluster import assign_candidate_group, cluster_within_groups
from .incidents import build_incidents, stitch_incidents


def run_pipeline(
    eve_json_path: str,
    out_dir: str = "out",
    model_name: str = DEFAULT_MODEL_NAME,
    limit: Optional[int] = None,
    window_minutes: Optional[int] = None,
    eps: float = 0.35,
    min_samples: int = 3,
    algo: str = "dbscan",
    distance_threshold: float = 0.35,
    xi: float = 0.05,
    min_cluster_size: int = 3,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    used_default_window = window_minutes is None
    window_minutes = 30 if window_minutes is None else window_minutes

    print("[1/6] Parsing Suricata alerts...")
    df = parse_suricata_alerts(eve_json_path, limit=limit)

    print("[2/6] Assigning candidate groups (time window + src_ip)...")
    df = assign_candidate_group(df, window_minutes=window_minutes)
    group_sizes = df["candidate_group"].value_counts()
    keep_groups = group_sizes[group_sizes >= min_samples].index
    df = df[df["candidate_group"].isin(keep_groups)].reset_index(drop=True)

    print(f"[2b/6] Kept {len(df)} alerts in candidate groups >= min_samples ({min_samples}).")

    print("[3/6] Embedding alerts with Hugging Face model...")
    embeddings = embed_alerts(df, model_name=model_name)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)

    print("[4/6] Clustering within candidate groups...")
    df = cluster_within_groups(
        df,
        embeddings,
        algo=algo,
        eps=eps,
        min_samples=min_samples,
        distance_threshold=distance_threshold,
        xi=xi,
        min_cluster_size=min_cluster_size,
    )

    print("[5/6] Building incidents...")
    incidents = build_incidents(df)
    incidents = stitch_incidents(incidents)

    # Map all cluster ids to their final stitched incident id
    cluster_to_incident = {}
    for inc in incidents:
        for cid in inc.get("merged_cluster_ids", [inc["cluster_id"]]):
            cluster_to_incident[int(cid)] = inc["incident_id"]

    df["incident_id"] = df["cluster_id"].apply(lambda cid: cluster_to_incident.get(int(cid)) if cid != -1 else None)

    print("[6/6] Writing outputs...")
    df_out = df.drop(columns=["ts_dt"])
    df_out.to_parquet(os.path.join(out_dir, "alerts_clustered.parquet"), index=False)

    with open(os.path.join(out_dir, "incidents.json"), "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)

    params = {
        "model": model_name,
        "algo": algo,
        "eps": eps,
        "distance_threshold": distance_threshold,
        "min_samples": min_samples,
    }
    if not used_default_window:
        params["window_minutes"] = window_minutes

    stats = {
        "total_alerts": int(len(df)),
        "clustered_alerts": int((df["cluster_id"] != -1).sum()),
        "noise_alerts": int((df["cluster_id"] == -1).sum()),
        "incidents": int(len(incidents)),
        "reduction_ratio_alerts_per_incident": float(len(df) / len(incidents)) if incidents else None,
        "params": params,
    }
    with open(os.path.join(out_dir, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Done.")
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Suricata alert clustering pipeline")
    ap.add_argument("--eve", required=True, help="Path to Suricata eve.json (JSON lines)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of alerts parsed (debug)")
    ap.add_argument("--algo", choices=["dbscan", "hierarchical", "optics"], default="dbscan", help="Clustering algorithm: dbscan, hierarchical, or optics")
    ap.add_argument("--xi", type=float, default=0.05, help="OPTICS xi steepness threshold (0–1). Only used when --algo optics")
    ap.add_argument("--min-cluster-size", type=int, default=3, help="OPTICS minimum cluster size. Only used when --algo optics")
    ap.add_argument("--dist-threshold", type=float, default=0.35, help="Hierarchical clustering distance threshold (cosine distance). Only used when --algo hierarchical")
    ap.add_argument("--window-min", type=int, default=None, help="Candidate grouping time window in minutes")
    ap.add_argument("--eps", type=float, default=0.35, help="DBSCAN eps (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples")
    args = ap.parse_args()

    run_pipeline(
        eve_json_path=args.eve,
        out_dir=args.out,
        model_name=args.model,
        limit=args.limit,
        window_minutes=args.window_min,
        eps=args.eps,
        min_samples=args.min_samples,
        algo=args.algo,
        distance_threshold=args.dist_threshold,
        xi=args.xi,
        min_cluster_size=args.min_cluster_size,
    )


if __name__ == "__main__":
    main()
