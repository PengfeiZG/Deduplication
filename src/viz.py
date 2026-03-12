import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUT_DIR = "out"
PARQUET_PATH = f"{OUT_DIR}/alerts_clustered.parquet"
EMBED_PATH = f"{OUT_DIR}/embeddings.npy"
STATS_PATH = f"{OUT_DIR}/run_stats.json"


def load_run_stats():
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_data():
    df = pd.read_parquet(PARQUET_PATH)
    emb = np.load(EMBED_PATH)

    mask = df["cluster_id"] != -1
    df = df.loc[mask].reset_index(drop=True)
    emb = emb[mask.values]

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df, emb


def plot_pca_scatter(df, emb, run_stats=None, max_points=20000):
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).reset_index(drop=True)
        emb = emb[df.index.values]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster_id"], s=6)

    title = "Alert clustering (PCA of sentence embeddings)"
    if run_stats and "params" in run_stats:
        p = run_stats["params"]
        title += f"\nmodel={p.get('model')} | window={p.get('window_minutes')} min | eps={p.get('eps')} | min_samples={p.get('min_samples')}"
    plt.title(title)

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    cbar = plt.colorbar(sc)
    cbar.set_label("Incident cluster ID (cluster_id)")
    plt.gcf().text(0.01, 0.01, "Each point = 1 Suricata IDS alert (noise alerts excluded).")

    plt.tight_layout()
    plt.show()


def plot_top_incidents(df, run_stats=None, top_n=20):
    label_col = "incident_id" if "incident_id" in df.columns and df["incident_id"].notna().any() else "cluster_id"
    counts = df[label_col].value_counts().head(top_n).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh([str(i) for i in counts.index], counts.values)

    title = f"Top {top_n} incidents by number of alerts"
    if run_stats:
        title += f"\n(total clustered alerts: {run_stats.get('clustered_alerts')}, incidents: {run_stats.get('incidents')})"
    plt.title(title)

    plt.xlabel("Number of alerts in incident")
    plt.ylabel("Incident ID" if label_col == "incident_id" else "Incident cluster ID (cluster_id)")
    plt.tight_layout()
    plt.show()


def plot_timeline(df, top_n=15):
    if df["ts"].isna().all():
        print("No valid timestamps found to plot timeline.")
        return

    key = "incident_id" if "incident_id" in df.columns and df["incident_id"].notna().any() else "cluster_id"
    top_ids = df[key].value_counts().head(top_n).index.tolist()
    d = df[df[key].isin(top_ids)].copy().dropna(subset=["ts"])

    plt.figure(figsize=(12, 6))
    plt.scatter(d["ts"], d[key], s=6)

    plt.title(f"Incident timeline (top {top_n} incidents by alert volume)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Incident ID" if key == "incident_id" else "Incident cluster ID (cluster_id)")
    plt.tight_layout()
    plt.show()


def show_incident_example(df, incident_label, k=12):
    if "incident_id" in df.columns and isinstance(incident_label, str):
        g = df[df["incident_id"] == incident_label].copy()
    else:
        g = df[df["cluster_id"] == int(incident_label)].copy()

    g = g.dropna(subset=["ts"]).sort_values("ts")

    cols = ["timestamp", "src_ip", "dest_ip", "dest_port", "proto", "signature", "category", "severity"]
    if "incident_id" in g.columns:
        cols = ["incident_id"] + cols
    print(g[cols].head(k).to_string(index=False))


def main():
    run_stats = load_run_stats()
    df, emb = load_data()

    plot_pca_scatter(df, emb, run_stats=run_stats)
    plot_top_incidents(df, run_stats=run_stats, top_n=25)
    plot_timeline(df, top_n=15)

    if "incident_id" in df.columns and df["incident_id"].notna().any():
        biggest = df["incident_id"].value_counts().idxmax()
        print(f"\nRepresentative alerts for incident_id={biggest}\n")
        show_incident_example(df, biggest, k=12)
    else:
        biggest = int(df["cluster_id"].value_counts().idxmax())
        print(f"\nRepresentative alerts for cluster_id={biggest}\n")
        show_incident_example(df, biggest, k=12)


if __name__ == "__main__":
    main()
