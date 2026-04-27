# ⚡ Suricata Alert Clustering

> AI-driven alert deduplication and incident clustering for Security Operations Centres.  
> Converts raw Suricata `eve.json` streams into prioritised, analyst-ready incidents — with no custom model training required.

---

## Results at a Glance

| Metric | Value |
|---|---|
| Alerts processed (test) | 100,000 |
| Incidents recovered | 20 / 20 expected |
| Alert → Incident compression | **4,010 : 1** |
| Noise alerts correctly isolated | ✓ |
| Custom model training required | **None** |

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Pipeline Parameters](#pipeline-parameters)
- [Clustering Algorithms](#clustering-algorithms)
- [Test Data](#test-data)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Architecture Notes](#architecture-notes)
- [Known Limitations](#known-limitations)

---

## Overview

Modern IDS deployments like Suricata generate hundreds of thousands of alerts per day. The vast majority are duplicate events from the same underlying attack campaign. This project uses **semantic embedding + unsupervised clustering** to automatically group related alerts into coherent incident cases, reducing analyst triage workload by orders of magnitude.

**No labelled training data. No custom model training. No infrastructure overhaul.**  
The system uses a pre-trained `sentence-transformers` model from Hugging Face as-is, and runs end-to-end on a single machine.

---

## How It Works

```
eve.json
   │
   ▼
[1] Parse & Normalize        parse_suricata.py
    Extract fields: timestamp, src_ip, dest_ip,
    ports, protocol, signature, category, severity

   ▼
[2] Feature Engineering      build_text.py
    Convert each alert to a structured text token string
    e.g. "SIG=ET BRUTE_FORCE SSH ... CAT=... SEV=2 PROTO=TCP DST_PORT=22"

   ▼
[3] Candidate Grouping       cluster.py
    Group alerts by src_ip + 30-minute time window
    (limits clustering scope — avoids O(n²) comparisons)

   ▼
[4] Semantic Embedding       embed.py
    Encode alert text → 384-dim vectors via MiniLM-L12-v2
    Results cached to disk by SHA-256 hash

   ▼
[5] Clustering               cluster.py
    Run DBSCAN / Hierarchical / OPTICS on cosine-distance
    embeddings within each candidate group

   ▼
[6] Incident Synthesis       incidents.py
    Assign family labels, risk scores, and natural-language
    summaries. Optionally stitch related clusters into
    unified long-running incidents.

   ▼
alerts_clustered.parquet  +  incidents.json  +  run_stats.json
```

---

## Project Structure

```
.
├── app.py                  # Streamlit dashboard
├── build_text.py           # Alert → token string feature engineering
├── cluster.py              # Candidate grouping + DBSCAN/Hierarchical/OPTICS
├── embed.py                # Sentence transformer embedding + disk cache
├── eval.py                 # Evaluation placeholder (purity / mixed-cluster rate)
├── incidents.py            # Incident synthesis, family labelling, risk scoring, stitching
├── parse_suricata.py       # Suricata EVE JSON parser
├── pipeline.py             # End-to-end orchestration + CLI entry point
├── hf_cache/               # Hugging Face model cache (auto-created, gitignored)
│   └── embeddings/         # Per-alert embedding cache (SHA-256 keyed .npy files)
├── out/                    # Default output directory
│   ├── alerts_clustered.parquet
│   ├── incidents.json
│   ├── embeddings.npy
│   └── run_stats.json
└── test_data/
    ├── test_eve.json        # Small synthetic dataset (965 alerts, 9 campaigns)
    └── large_test_eve.json  # Large synthetic dataset (100k alerts, 20 campaigns)
```

---

## Installation

**Python 3.10+ required.**

```bash
# 1. Clone the repository
git clone https://github.com/your-username/suricata-alert-clustering.git
cd suricata-alert-clustering

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
streamlit>=1.35
pandas>=2.0
numpy>=1.26
pyarrow>=15.0
scikit-learn>=1.4
sentence-transformers>=3.0
tqdm>=4.66
plotly>=5.20
umap-learn>=0.5          # Optional — for UMAP projection in dashboard
```

> **Note:** The first run downloads the `all-MiniLM-L12-v2` model (~130 MB) into `hf_cache/`.  
> Subsequent runs use the local cache and are significantly faster.

---

## Quickstart

### CLI — run the full pipeline

```bash
# Run on the small test dataset
python -m pipeline --eve test_data/test_eve.json --out out/

# Run on the large test dataset
python -m pipeline --eve test_data/large_test_eve.json --out out/

# Custom parameters
python -m pipeline \
  --eve /path/to/eve.json \
  --out out/ \
  --algo dbscan \
  --eps 0.35 \
  --min-samples 5 \
  --window-min 30
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Streamlit Dashboard

The dashboard provides a full analyst workflow in five tabs:

| Tab | Description |
|---|---|
| **Incidents** | Expandable incident cards with all alerts, signatures, timelines, IPs, ports, risk scores, and family labels. Filterable by severity, keyword, sort order. |
| **Alert Table** | Full clustered alert table with search and noise toggle. |
| **Cluster Map** | UMAP / PCA / t-SNE projection of alert embeddings. Colour by cluster, severity, protocol, category, or source IP. Interactive hover with alert detail. |
| **Distributions** | Alerts per cluster, top signatures, severity breakdown, top source IPs. |
| **Download** | Export `alerts_clustered.parquet`, `incidents.json`, and `run_stats.json`. |

### Dashboard configuration

Place a `.streamlit/config.toml` file in the same directory as `app.py`:

```toml
[server]
maxUploadSize = 1024      # MB — supports files up to 1 GB
maxMessageSize = 1024
```

---

## Pipeline Parameters

| Parameter | Default | Description |
|---|---|---|
| `--eve` | *(required)* | Path to Suricata `eve.json` log file |
| `--out` | `out/` | Output directory |
| `--model` | `all-MiniLM-L12-v2` | Hugging Face sentence transformer model |
| `--algo` | `dbscan` | Clustering algorithm: `dbscan`, `hierarchical`, `optics` |
| `--eps` | `0.35` | DBSCAN cosine distance threshold |
| `--min-samples` | `5` | Minimum alerts to form a cluster |
| `--window-min` | `30` | Candidate group time window in minutes |
| `--dist-threshold` | `0.35` | Hierarchical clustering distance threshold |
| `--xi` | `0.05` | OPTICS steepness threshold |
| `--min-cluster-size` | `3` | OPTICS minimum cluster size |
| `--limit` | *None* | Cap alert count (useful for debug runs) |

---

## Clustering Algorithms

### DBSCAN *(default)*
Density-based clustering with a fixed `eps` radius. Best for datasets where clusters have roughly uniform density. Alerts outside any cluster are automatically labelled as noise (`cluster_id = -1`).

```bash
python -m pipeline --eve eve.json --algo dbscan --eps 0.35 --min-samples 5
```

### Hierarchical (Agglomerative)
Average-linkage clustering with a cosine distance threshold. Produces no explicit noise label — all alerts are assigned a cluster, with small clusters post-processed out by `min_samples`.

```bash
python -m pipeline --eve eve.json --algo hierarchical --dist-threshold 0.35 --min-samples 5
```

### OPTICS
Variable-density extension of DBSCAN. Best for datasets with campaigns of mixed alert rates (e.g., a slow C2 beacon alongside a high-volume port scan). No `eps` required.

```bash
python -m pipeline --eve eve.json --algo optics --xi 0.05 --min-cluster-size 5
```

---

## Test Data

Two synthetic `eve.json` files are included for validation:

### `test_eve.json` — Small dataset
- **965 alerts** · **9 campaigns** · **~280 noise alerts**
- Campaigns: SSH brute-force, RDP brute-force, Nmap scan, ICMP sweep, Log4Shell, SQL injection, Cobalt Strike C2, SMB lateral movement, DNS tunnelling
- Expected result: **9 incidents**

### `large_test_eve.json` — Large dataset
- **100,000 alerts** · **20 campaigns** · **~21,000 noise alerts**
- All 20 campaigns start exactly at 30-minute window boundaries and finish within 25 minutes, guaranteeing no campaign crosses a window boundary
- Expected result: **20 incidents**

| Campaign | src_ip | Alerts | Severity |
|---|---|---|---|
| SSH brute-force | 192.168.10.5 | 6,000 | 2 |
| RDP brute-force | 192.168.10.6 | 5,000 | 2 |
| FTP brute-force | 192.168.10.7 | 2,500 | 2 |
| Telnet brute-force | 192.168.10.8 | 2,000 | 2 |
| Nmap TCP port scan | 192.168.20.1 | 10,000 | 2 |
| ICMP ping sweep | 192.168.20.2 | 1,500 | 3 |
| UDP service discovery | 192.168.20.3 | 1,200 | 3 |
| Log4Shell exploit | 185.220.101.1 | 4,000 | 1 |
| SQL injection | 185.220.101.5 | 6,000 | 1 |
| XSS / web app attack | 185.220.101.6 | 3,000 | 2 |
| Cobalt Strike C2 | 10.0.0.77 | 8,000 | 1 |
| Emotet C2 callbacks | 10.0.0.78 | 5,000 | 1 |
| SMB / EternalBlue | 10.0.0.20 | 6,000 | 1 |
| WMI / DCOM lateral | 10.0.0.21 | 2,500 | 1 |
| DNS tunnelling / DGA | 192.168.30.1 | 4,000 | 2 |
| Data exfiltration | 10.0.0.88 | 3,000 | 2 |
| Tor anonymisation | 10.0.0.99 | 2,000 | 2 |
| Cryptomining | 10.0.0.111 | 2,500 | 2 |
| Shellshock RCE | 45.33.32.156 | 3,000 | 1 |
| PrintNightmare | 10.0.0.22 | 3,000 | 1 |

---

## Output Files

### `incidents.json`
Array of incident objects, sorted by start time and descending risk score.

```json
{
  "incident_id": "INC-00000",
  "cluster_id": 0,
  "start_time": "2024-06-01T00:01:00",
  "end_time": "2024-06-01T00:24:59",
  "src_ips": ["10.0.0.77"],
  "dest_ips": ["91.108.4.1"],
  "dest_ports": [443],
  "proto": ["TCP"],
  "alert_count": 8000,
  "top_signatures": { "ET MALWARE Cobalt Strike Beacon Detected": 5334 },
  "max_severity": 1,
  "family": "malware_c2",
  "risk_score": 90,
  "summary": "8000 alerts indicate repeated malware command-and-control or beaconing activity from 10.0.0.77 to 91.108.4.1 on ports 443.",
  "merged_cluster_ids": [0]
}
```

### `alerts_clustered.parquet`
Full alert table with added columns:

| Column | Description |
|---|---|
| `cluster_id` | Cluster assignment (`-1` = noise) |
| `incident_id` | Mapped incident ID (e.g. `INC-00000`) |
| `candidate_group` | src_ip + time window bucket used for grouping |

### `run_stats.json`
Pipeline summary statistics.

```json
{
  "total_alerts": 80200,
  "clustered_alerts": 80200,
  "noise_alerts": 0,
  "incidents": 20,
  "reduction_ratio_alerts_per_incident": 4010.0,
  "params": {
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "algo": "dbscan",
    "eps": 0.35,
    "min_samples": 5
  }
}
```

---

## Configuration

### Embedding model

Three pre-trained models are supported out of the box. All are sourced from the official Hugging Face `sentence-transformers` organisation.

| Model | Dim | Size | Notes |
|---|---|---|---|
| `all-MiniLM-L12-v2` *(default)* | 384 | ~130 MB | Best balance of speed and accuracy for short structured text |
| `all-MiniLM-L6-v2` | 384 | ~90 MB | Faster, slightly lower accuracy |
| `all-mpnet-base-v2` | 768 | ~420 MB | Highest accuracy, significantly slower |

> **Important:** Switching models invalidates the embedding cache. Delete `hf_cache/embeddings/` before switching, or the pipeline will silently mix embeddings from different models (they have incompatible shapes for mpnet).

### Tuning the time window

The `--window-min` parameter controls how far apart in time two alerts from the same source IP can be while still being considered for clustering together. Longer windows allow detection of slow, persistent campaigns at the cost of more candidate pairs.

| Use case | Recommended window |
|---|---|
| High-volume scans / brute-force | 15–30 min |
| Slow C2 beaconing | 30–60 min |
| DGA / DNS tunnelling | 30–45 min |
| Default (balanced) | **30 min** |

### Tuning `eps` for DBSCAN

`eps` is the maximum cosine distance between two alert embeddings for them to be considered neighbours. Alerts with very similar signatures, ports, and protocols will have cosine distances near 0.0; unrelated alerts approach 1.0.

| Situation | Action |
|---|---|
| Too many small clusters | Increase `eps` (try 0.40–0.50) |
| Campaigns merging together | Decrease `eps` (try 0.25–0.30) |
| Scans fragmenting | Increase `eps` or use OPTICS |

---

## Architecture Notes

### Why candidate grouping?

Clustering all 100,000 alerts together would require a 100k × 100k distance matrix — computationally infeasible. By pre-filtering on `src_ip + time_window`, each clustering call operates on a few hundred to a few thousand alerts, making the pipeline scale linearly with alert volume.

### Why remove `SRC_PORT` from alert text?

Source ports are ephemeral — randomised by the OS per connection. Including them in the embedding would cause alerts from the same campaign (same src_ip, same destination, same signature) to appear semantically dissimilar. Only the `SRC_BUCKET` (semantic category: `admin`, `web`, `dns`, `mail`, `smb`, `other`) is retained.

### Incident stitching

After clustering, `stitch_incidents()` in `incidents.py` performs a second-pass merge of clusters that share the same attack family, source IP, and are separated by fewer than 30 minutes. This handles long-running campaigns that span multiple 30-minute candidate group windows. Rules are conservative to avoid false merges.

### Embedding cache

Each alert text is hashed (SHA-256) and its embedding saved as a `.npy` file under `hf_cache/embeddings/`. On subsequent runs, cached embeddings are loaded from disk instead of re-encoded. This makes re-runs with different clustering parameters near-instant after the first full embedding pass.

---

## Known Limitations

- **Synthetic test data only.** Results on the 100k synthetic dataset are strong. Performance on real-world enterprise telemetry with mixed benign/malicious traffic has not yet been measured.
- **Single-machine.** The pipeline runs in-process on a single machine. For production-scale (millions of alerts/day), the embedding and clustering stages would need to be distributed.
- **30-minute window is fixed per run.** Long-duration APT campaigns that are active for hours may be split across multiple candidate group windows and require the stitching layer to reunite them. Very slow campaigns (one alert per hour) will not cluster at all.
- **UMAP visual ≠ clustering truth.** The UMAP projection is a lossy 2D approximation of 384-dimensional embedding space. Clusters that appear scattered on the map may still be correctly identified by DBSCAN in high-dimensional space.

---

## `.gitignore` Recommendations

```gitignore
# Hugging Face model weights
hf_cache/

# Pipeline output
out/

# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Streamlit
.streamlit/secrets.toml

# Large test files (optional — add if you don't want them in the repo)
# test_data/large_test_eve.json
```

---

## Acknowledgements

- [Suricata IDS](https://suricata.io/) — open-source network threat detection engine
- [sentence-transformers](https://www.sbert.net/) — semantic embedding library by UKP Lab
- [scikit-learn](https://scikit-learn.org/) — DBSCAN, Agglomerative Clustering, OPTICS, PCA, t-SNE
- [UMAP](https://umap-learn.readthedocs.io/) — Uniform Manifold Approximation and Projection
- [Streamlit](https://streamlit.io/) — dashboard framework
- [Plotly](https://plotly.com/python/) — interactive visualisation
