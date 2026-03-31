"""
Suricata Alert Clustering — Streamlit GUI
Run with: streamlit run app.py
"""

import importlib.util
import json
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ── Package loader ────────────────────────────────────────────────────────────
# The project modules use relative imports (e.g. `from .build_text import …`).
# Load them as a synthetic in-memory package so those imports resolve correctly
# regardless of where app.py is placed on disk.

def _load_project_as_package(pkg_dir: str, pkg_name: str = "_suricata_pkg"):
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [pkg_dir]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    for mod_name in ["build_text", "parse_suricata", "embed", "cluster", "incidents", "pipeline"]:
        fpath = os.path.join(pkg_dir, f"{mod_name}.py")
        if not os.path.exists(fpath):
            continue
        full_name = f"{pkg_name}.{mod_name}"
        spec = importlib.util.spec_from_file_location(
            full_name, fpath, submodule_search_locations=[]
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, mod_name, mod)

    return pkg

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Suricata Alert Clustering",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0a0c10;
        color: #c8d0db;
    }

    /* Streamlit app background */
    .stApp {
        background: #0a0c10;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1318 !important;
        border-right: 1px solid #1e2530;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: #7a8799 !important;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    [data-testid="stSidebar"] h3 {
        color: #e0e8f0 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid #1e2530;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Main header */
    .main-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        line-height: 1.2;
        margin-bottom: 2px;
    }
    .main-subheader {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #3d5066;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 24px;
    }

    /* Stat cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 28px;
    }
    .stat-card {
        background: #0f1318;
        border: 1px solid #1e2530;
        border-top: 2px solid #00d4ff;
        padding: 16px 18px;
        border-radius: 2px;
    }
    .stat-card.orange { border-top-color: #FFB800; }   /* amber  */
    .stat-card.red    { border-top-color: #BF5AF2; }   /* purple */
    .stat-card.green  { border-top-color: #30D158; }   /* green  */
    .stat-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e8f0f8;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #4a6070;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Section labels */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3d5066;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        border-bottom: 1px solid #1a2028;
        padding-bottom: 6px;
        margin-bottom: 16px;
        margin-top: 28px;
    }

    /* Incident card */
    .incident-card {
        background: #0f1318;
        border: 1px solid #1a2530;
        border-left: 3px solid #00d4ff;
        padding: 14px 18px;
        border-radius: 2px;
        margin-bottom: 10px;
    }
    .incident-card.sev-high   { border-left-color: #ff2d55; }   /* vivid red        */
    .incident-card.sev-medium { border-left-color: #ffb800; }   /* vivid amber       */
    .incident-card.sev-low    { border-left-color: #30d158; }   /* vivid green       */
    .incident-card.sev-info   { border-left-color: #64d2ff; }   /* sky blue          */
    .inc-id {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 0.08em;
    }
    .inc-summary {
        font-size: 0.88rem;
        color: #8a9db0;
        margin-top: 5px;
        line-height: 1.5;
    }
    .inc-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3d5066;
        margin-top: 8px;
        display: flex;
        gap: 20px;
    }
    .inc-badge {
        display: inline-block;
        background: #1a2530;
        color: #4a8fa8;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        padding: 2px 7px;
        border-radius: 2px;
        margin-right: 4px;
        letter-spacing: 0.06em;
    }
    .inc-badge.sev1  { background: #2d1018; color: #ff2d55; }   /* red   — sev 1    */
    .inc-badge.sev2  { background: #2d2000; color: #ffb800; }   /* amber — sev 2    */
    .inc-badge.sev3  { background: #0d2018; color: #30d158; }   /* green — sev 3    */
    .inc-badge.sev4  { background: #0d1e28; color: #64d2ff; }   /* blue  — sev 4    */

    /* Pipeline log */
    .log-box {
        background: #080b0f;
        border: 1px solid #1a2028;
        border-radius: 2px;
        padding: 14px 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #4a8060;
        max-height: 260px;
        overflow-y: auto;
        white-space: pre-wrap;
        line-height: 1.7;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1e2e3a !important;
        border-radius: 2px !important;
        background: #080b0f !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        background: #00d4ff !important;
        color: #000 !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        transition: background 0.15s;
    }
    .stButton > button:hover {
        background: #00b8e0 !important;
    }
    .stButton > button:disabled {
        background: #1a2530 !important;
        color: #3d5066 !important;
    }

    /* Slider and number input */
    .stSlider > div > div > div { background: #00d4ff !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #1a2530 !important;
        border-radius: 2px !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        color: #4a6070 !important;
        background: #0f1318 !important;
    }

    /* Tab */
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #3d5066;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #e0e8f0;
    }

    /* Radio */
    .stRadio label { color: #7a8799 !important; font-size: 0.83rem !important; }

    /* Alerts / warnings */
    .stAlert { border-radius: 2px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">⚡ Suricata IDS · Alert Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subheader">AI-Driven Alert Deduplication and Incident Clustering</div>', unsafe_allow_html=True)

# ── Sidebar — Pipeline Configuration ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Pipeline Config")

    uploaded_file = st.file_uploader(
        "eve.json file",
        type=["json"],
        help="Suricata EVE JSON log (newline-delimited JSON)",
    )

    st.markdown("---")
    st.markdown("### 🔬 Grouping")

    window_minutes = st.slider("Time window (minutes)", 1, 60, 10, 1,
        help="Alerts from the same src_ip within this window are candidate cluster members.")
    min_samples = st.slider("Min cluster size", 2, 20, 3, 1,
        help="Minimum alerts required to form a cluster.")

    st.markdown("---")
    st.markdown("### 🧮 Clustering")

    algo = st.radio(
        "Algorithm",
        ["dbscan", "hierarchical", "optics"],
        index=0,
        help=(
            "DBSCAN: density-based, requires ε and min_samples.\n"
            "Hierarchical: agglomerative average-link, requires distance threshold.\n"
            "OPTICS: variable-density extension of DBSCAN — no ε required, "
            "handles clusters of varying density automatically."
        ),
    )

    if algo == "dbscan":
        eps = st.slider("DBSCAN ε (cosine dist)", 0.05, 0.80, 0.25, 0.01,
            help="Max cosine distance between neighbouring points.")
        dist_threshold = 0.25
        xi            = 0.05
        min_cluster_size = min_samples
    elif algo == "optics":
        xi = st.slider("OPTICS ξ (xi)", 0.01, 0.50, 0.05, 0.01,
            help="Steepness threshold for cluster boundary detection (0–0.5). "
                 "Lower values create more, tighter clusters.")
        min_cluster_size = st.slider("Min cluster size", 2, 20, min_samples, 1,
            help="Minimum number of alerts to form an OPTICS cluster. "
                 "Defaults to the global min_samples value above.")
        eps           = 0.25
        dist_threshold = 0.25
    else:
        dist_threshold = st.slider("Distance threshold (cosine)", 0.05, 0.80, 0.25, 0.01,
            help="Agglomerative clustering cut threshold.")
        eps           = 0.25
        xi            = 0.05
        min_cluster_size = min_samples

    st.markdown("---")
    st.markdown("### 🤖 Embedding Model")

    model_name = st.text_input(
        "Sentence Transformer",
        value="sentence-transformers/all-MiniLM-L6-v2",
        disabled=True
    )

    limit = st.number_input("Alert parse limit (0 = all)", min_value=0, value=0, step=500,
        help="Cap alerts for faster debug runs.")
    limit = int(limit) if limit > 0 else None

    st.markdown("---")
    run_btn = st.button("▶  Run Pipeline", use_container_width=True, disabled=uploaded_file is None)

# ── Helpers ───────────────────────────────────────────────────────────────────

def sev_class(sev: Optional[int]) -> str:
    if sev is None:
        return ""
    if sev == 1:
        return "sev-high"
    if sev == 2:
        return "sev-medium"
    if sev == 3:
        return "sev-low"
    return "sev-info"


def render_stat_cards(stats: dict):
    clustered_pct = (
        100 * stats["clustered_alerts"] / stats["total_alerts"]
        if stats["total_alerts"] else 0
    )
    ratio = stats.get("reduction_ratio_alerts_per_incident")
    ratio_str = f"{ratio:.1f}×" if ratio else "—"

    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card">
            <div class="stat-val">{stats['total_alerts']:,}</div>
            <div class="stat-label">Total Alerts</div>
          </div>
          <div class="stat-card orange">
            <div class="stat-val">{stats['incidents']:,}</div>
            <div class="stat-label">Incidents</div>
          </div>
          <div class="stat-card red">
            <div class="stat-val">{clustered_pct:.0f}%</div>
            <div class="stat-label">Clustered</div>
          </div>
          <div class="stat-card green">
            <div class="stat-val">{ratio_str}</div>
            <div class="stat-label">Alert → Incident Ratio</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_incident_card(inc: dict):
    sev = inc.get("max_severity")
    cls = sev_class(sev)
    _SEV_LABELS = {1: ("sev1", "CRITICAL"), 2: ("sev2", "HIGH"), 3: ("sev3", "MEDIUM"), 4: ("sev4", "LOW")}
    _sev_cls, _sev_label = _SEV_LABELS.get(sev, ("", "")) if sev else ("", "")
    sev_badge = f'<span class="inc-badge {_sev_cls}">SEV {sev} · {_sev_label}</span>' if sev else ""
    alert_count = inc["alert_count"]
    start = inc["start_time"][:19].replace("T", " ")
    end   = inc["end_time"][:19].replace("T", " ")

    top_sigs_html = " ".join(
        f'<span class="inc-badge">{sig[:48]}</span>'
        for sig in list(inc["top_signatures"].keys())[:2]
    )

    src_ips = ", ".join(inc["src_ips"][:3]) + ("…" if len(inc["src_ips"]) > 3 else "")
    ports   = ", ".join(map(str, inc["dest_ports"][:6])) + ("…" if len(inc["dest_ports"]) > 6 else "")

    st.markdown(
        f"""
        <div class="incident-card {cls}">
          <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <span class="inc-id">{inc['incident_id']}</span>
            <span>{sev_badge} <span class="inc-badge">{alert_count} alerts</span></span>
          </div>
          <div class="inc-summary">{inc['summary']}</div>
          <div style="margin-top:8px">{top_sigs_html}</div>
          <div class="inc-meta">
            <span>🕐 {start} → {end}</span>
            <span>🔌 ports {ports or "—"}</span>
            <span>🖥 {src_ips}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline_ui(
    eve_path: str,
    out_dir: str,
    model_name: str,
    limit,
    window_minutes: int,
    eps: float,
    min_samples: int,
    algo: str,
    distance_threshold: float,
    xi: float,
    min_cluster_size: int,
    log_placeholder,
):
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        log_placeholder.markdown(
            '<div class="log-box">' + "\n".join(log_lines) + "</div>",
            unsafe_allow_html=True,
        )

    try:
        # Load project modules as a synthetic package so relative imports work.
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        pkg = _load_project_as_package(pkg_dir)

        parse_suricata_alerts   = pkg.parse_suricata.parse_suricata_alerts
        embed_alerts            = pkg.embed.embed_alerts
        assign_candidate_group  = pkg.cluster.assign_candidate_group
        cluster_within_groups   = pkg.cluster.cluster_within_groups
        build_incidents         = pkg.incidents.build_incidents

        log("→ [1/6] Parsing Suricata alerts...")
        df = parse_suricata_alerts(eve_path, limit=limit)
        log(f"   ✓ Parsed {len(df):,} alert events.")

        log(f"→ [2/6] Assigning candidate groups (window={window_minutes}m)...")
        df = assign_candidate_group(df, window_minutes=window_minutes)
        group_sizes = df["candidate_group"].value_counts()
        keep_groups = group_sizes[group_sizes >= min_samples].index
        df = df[df["candidate_group"].isin(keep_groups)].reset_index(drop=True)
        log(f"   ✓ {len(df):,} alerts in viable groups (≥{min_samples}).")

        log(f"→ [3/6] Embedding with '{model_name}'...")
        embeddings = embed_alerts(df, model_name=model_name)
        np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
        log(f"   ✓ Embeddings shape: {embeddings.shape}.")

        log(f"→ [4/6] Clustering ({algo}, eps={eps}, min_samples={min_samples})...")
        df = cluster_within_groups(
            df, embeddings,
            algo=algo, eps=eps, min_samples=min_samples,
            distance_threshold=distance_threshold,
            xi=xi,
            min_cluster_size=min_cluster_size,
        )
        n_clusters = int((df["cluster_id"] != -1).sum())
        log(f"   ✓ {n_clusters:,} alerts assigned to clusters.")

        log("→ [5/6] Building incidents...")
        incidents = build_incidents(df)
        cluster_to_incident = {inc["cluster_id"]: inc["incident_id"] for inc in incidents}
        df["incident_id"] = df["cluster_id"].apply(
            lambda cid: cluster_to_incident.get(int(cid)) if cid != -1 else None
        )
        log(f"   ✓ {len(incidents):,} incidents generated.")

        log("→ [6/6] Writing outputs...")
        df_out = df.drop(columns=["ts_dt"], errors="ignore")
        df_out.to_parquet(os.path.join(out_dir, "alerts_clustered.parquet"), index=False)
        with open(os.path.join(out_dir, "incidents.json"), "w") as f:
            json.dump(incidents, f, indent=2)

        stats = {
            "total_alerts": int(len(df)),
            "clustered_alerts": int((df["cluster_id"] != -1).sum()),
            "noise_alerts": int((df["cluster_id"] == -1).sum()),
            "incidents": int(len(incidents)),
            "reduction_ratio_alerts_per_incident": (
                float(len(df) / len(incidents)) if incidents else None
            ),
        }
        with open(os.path.join(out_dir, "run_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        log("")
        log("✅ Pipeline complete.")
        return df_out, incidents, stats, embeddings

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        log(traceback.format_exc())
        return None, None, None, None


# ── Main area ─────────────────────────────────────────────────────────────────

# Session state
if "results" not in st.session_state:
    st.session_state.results = None

if not uploaded_file:
    st.markdown(
        """
        <div style="background:#0f1318; border:1px dashed #1e2e3a; border-radius:2px;
                    padding:40px; text-align:center; margin-top:20px;">
          <div style="font-family:'IBM Plex Mono',monospace; color:#2a4055;
                      font-size:0.9rem; letter-spacing:0.12em; text-transform:uppercase;">
            Upload an eve.json file and configure the pipeline in the sidebar,<br>
            then click <strong style="color:#3d6080">▶ Run Pipeline</strong> to begin.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    # Clean up any previous run's temp directory before starting a new one.
    prev_tmp = st.session_state.get("_tmp_dir")
    if prev_tmp and os.path.isdir(prev_tmp):
        import shutil
        shutil.rmtree(prev_tmp, ignore_errors=True)

    # Use a persistent temp dir (NOT a context manager) so the parquet file
    # remains on disk until the user downloads it or starts the next run.
    tmp = tempfile.mkdtemp(prefix="suricata_gui_")
    st.session_state["_tmp_dir"] = tmp

    eve_path = os.path.join(tmp, "eve.json")
    out_dir  = os.path.join(tmp, "out")
    os.makedirs(out_dir)

    # Stream to disk in 256 MB chunks — handles files up to 3 GB without
    # loading the entire buffer into Python memory at once.
    CHUNK = 256 * 1024 * 1024  # 256 MB
    uploaded_file.seek(0)
    with open(eve_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)

    st.markdown('<div class="section-label">Pipeline Execution Log</div>', unsafe_allow_html=True)
    log_ph = st.empty()

    df_out, incidents, stats, embeddings = run_pipeline_ui(
        eve_path=eve_path,
        out_dir=out_dir,
        model_name=model_name,
        limit=limit,
        window_minutes=window_minutes,
        eps=eps,
        min_samples=min_samples,
        algo=algo,
        distance_threshold=dist_threshold,
        xi=xi,
        min_cluster_size=min_cluster_size,
        log_placeholder=log_ph,
    )

    if df_out is not None:
        parquet_path = os.path.join(out_dir, "alerts_clustered.parquet")
        st.session_state.results = {
            "df": df_out,
            "incidents": incidents,
            "stats": stats,
            "embeddings": embeddings,
            "incidents_json": json.dumps(incidents, indent=2),
            "parquet_path": parquet_path,
        }

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.results:
    r = st.session_state.results
    df         = r["df"]
    incidents  = r["incidents"]
    stats      = r["stats"]

    # Stat cards
    st.markdown('<div class="section-label">Run Summary</div>', unsafe_allow_html=True)
    render_stat_cards(stats)

    # Tabs
    tab_inc, tab_alerts, tab_cluster, tab_dist, tab_download = st.tabs([
        "Incidents", "Alert Table", "Cluster Map", "Distributions", "Download"
    ])

    # ── Incidents tab ─────────────────────────────────────────────────────────
    with tab_inc:
        if not incidents:
            st.info("No incidents formed. Try lowering min_samples or increasing the time window.")
        else:
            # ── Filter controls ───────────────────────────────────────────
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search = st.text_input(
                    "Search incidents",
                    placeholder="signature, IP, port, category, incident ID…",
                    label_visibility="collapsed",
                )
            with col2:
                sev_filter = st.selectbox(
                    "Severity",
                    ["All", "1 (High)", "2 (Medium)", "3 (Low)", "Unknown"],
                    label_visibility="collapsed",
                    help="Show only incidents whose max severity matches exactly. Suricata: 1 = highest, 3 = lowest.",
                )
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Time (asc)", "Alert count (desc)", "Severity (asc)", "Duration (desc)"],
                    label_visibility="collapsed",
                )

            # ── Apply filters ─────────────────────────────────────────────────
            filtered = incidents.copy()

            # Text search — covers all analyst-relevant fields
            if search:
                s = search.lower().strip()
                def _inc_matches(i):
                    # summary, signatures, IPs
                    if s in i["summary"].lower():
                        return True
                    if any(s in sig.lower() for sig in i["top_signatures"]):
                        return True
                    if any(s in ip for ip in i["src_ips"] + i["dest_ips"]):
                        return True
                    # incident_id (e.g. "INC-00003")
                    if s in i["incident_id"].lower():
                        return True
                    # dest_ports — match exact port number as token
                    if any(s == str(p) for p in i.get("dest_ports", [])):
                        return True
                    # proto
                    if any(s in pr.lower() for pr in i.get("proto", [])):
                        return True
                    return False
                filtered = [i for i in filtered if _inc_matches(i)]

            # Severity filter — explicit logic, no silent None→99 substitution
            if sev_filter == "1 (High)":
                filtered = [i for i in filtered if i.get("max_severity") == 1]
            elif sev_filter == "2 (Medium)":
                filtered = [i for i in filtered if i.get("max_severity") == 2]
            elif sev_filter == "3 (Low)":
                filtered = [i for i in filtered if i.get("max_severity") == 3]
            elif sev_filter == "Unknown":
                filtered = [i for i in filtered if i.get("max_severity") is None]
            # else "All" → no filter applied

            # Sort — None severity always goes to the end regardless of direction
            if sort_by == "Alert count (desc)":
                filtered = sorted(filtered, key=lambda x: -x["alert_count"])
            elif sort_by == "Severity (asc)":
                filtered = sorted(filtered, key=lambda x: (x.get("max_severity") is None, x.get("max_severity") or 0))
            elif sort_by == "Duration (desc)":
                def _duration(x):
                    try:
                        from datetime import datetime
                        s_ = datetime.fromisoformat(x["start_time"])
                        e_ = datetime.fromisoformat(x["end_time"])
                        return -((e_ - s_).total_seconds())
                    except Exception:
                        return 0
                filtered = sorted(filtered, key=_duration)
            # else "Time (asc)" — already sorted by time from build_incidents

            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d5066;margin-bottom:12px;">'
                f'SHOWING {len(filtered)} / {len(incidents)} INCIDENTS</div>',
                unsafe_allow_html=True,
            )

            for inc in filtered:
                render_incident_card(inc)
                with st.expander(f"All alerts ({inc['alert_count']})"):
                    # Pull every alert for this cluster directly from df,
                    # not the 5-row representative_alerts snapshot in the incident dict.
                    cols = [c for c in [
                        "timestamp", "src_ip", "src_port", "dest_ip", "dest_port",
                        "proto", "signature", "category", "severity",
                    ] if c in df.columns]
                    all_alerts = df[df["cluster_id"] == inc["cluster_id"]][cols].copy()
                    all_alerts = all_alerts.sort_values("timestamp").reset_index(drop=True)
                    st.dataframe(all_alerts, use_container_width=True, hide_index=True)

    # ── Alert table tab ───────────────────────────────────────────────────────
    with tab_alerts:
        cols_show = [c for c in [
            "timestamp", "src_ip", "src_port", "dest_ip", "dest_port",
            "proto", "signature", "category", "severity",
            "cluster_id", "incident_id",
        ] if c in df.columns]

        col1, col2 = st.columns([3, 1])
        with col1:
            q = st.text_input("Filter alerts", placeholder="IP, signature…", label_visibility="collapsed")
        with col2:
            show_noise = st.checkbox("Show unclassified (noise)", value=False)

        disp = df[cols_show].copy()
        if not show_noise:
            disp = disp[disp["cluster_id"] != -1]
        if q:
            q_clean = q.lower().strip()
            # Search text columns as substrings; search numeric columns as exact token matches
            # to avoid "22" matching port 2200 or cluster_id 220.
            text_cols = [c for c in cols_show if disp[c].dtype == object]
            num_cols  = [c for c in cols_show if disp[c].dtype != object]
            text_mask = disp[text_cols].astype(str).apply(
                lambda col: col.str.lower().str.contains(q_clean, na=False, regex=False)
            ).any(axis=1)
            num_mask = disp[num_cols].astype(str).apply(
                lambda col: col.str.lower() == q_clean
            ).any(axis=1) if num_cols else pd.Series(False, index=disp.index)
            disp = disp[text_mask | num_mask]

        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d5066;margin-bottom:8px;">'
            f'{len(disp):,} alerts displayed</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(disp.head(2000), use_container_width=True, hide_index=True, height=450)
        if len(disp) > 2000:
            st.caption(f"Displaying first 2,000 of {len(disp):,}. Download the full parquet for all rows.")


    # ── Cluster Map tab ───────────────────────────────────────────────────────
    with tab_cluster:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            has_plotly = True
        except ImportError:
            has_plotly = False
            st.warning("Install plotly for the cluster map: pip install plotly")

        embeddings = r.get("embeddings")

        if not has_plotly:
            pass
        elif embeddings is None or len(embeddings) == 0:
            st.info("No embeddings available — re-run the pipeline.")
        else:
            # ── Projection (cached per run) ───────────────────────────────────
            proj_key = "_proj_2d"
            proj_method_key = "_proj_method"

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
            with col_ctrl1:
                proj_method = st.radio(
                    "Projection method",
                    ["UMAP", "PCA", "t-SNE"],
                    horizontal=True,
                    help=(
                        "UMAP: best cluster separation, requires umap-learn. "
                        "PCA: instant, linear. "
                        "t-SNE: good local structure, slow on large datasets."
                    ),
                )
            with col_ctrl2:
                show_noise_map = st.checkbox("Show noise points", value=True,
                    help="Alerts with cluster_id = -1 shown as grey dots.")
            with col_ctrl3:
                color_by = st.selectbox(
                    "Colour by",
                    ["cluster_id", "severity", "proto", "category", "src_ip"],
                    help=(
                        "cluster_id — one colour per detected cluster.  "
                        "severity — red=1 (high) → teal=3 (low).  "
                        "proto — TCP/UDP/etc.  "
                        "category — Suricata alert category.  "
                        "src_ip — top 15 source IPs, rest grouped as 'other'."
                    ),
                )

            # Re-project if method changed or no cache
            needs_reproject = (
                proj_key not in st.session_state
                or st.session_state.get(proj_method_key) != proj_method
            )

            if needs_reproject:
                with st.spinner(f"Projecting {len(embeddings):,} embeddings with {proj_method}…"):
                    try:
                        _pca_var = None   # only set for PCA projections
                        _actual_method = proj_method  # track what was actually run

                        if proj_method == "UMAP":
                            import umap
                            reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
                            coords = reducer.fit_transform(embeddings)

                        elif proj_method == "t-SNE":
                            from sklearn.decomposition import PCA as _PCA
                            from sklearn.manifold import TSNE
                            pca_init = _PCA(n_components=2).fit_transform(embeddings)
                            if len(embeddings) > 5000:
                                # t-SNE is O(n log n); above 5k use PCA and note it clearly
                                coords = pca_init
                                _actual_method = "PCA"
                                _pca_var = _PCA(n_components=2).fit(embeddings).explained_variance_ratio_ * 100
                                st.info(
                                    f"⚠ Dataset has {len(embeddings):,} alerts — t-SNE would be too slow. "
                                    "PCA projection shown instead. Use UMAP for better separation on large datasets."
                                )
                            else:
                                coords = TSNE(
                                    n_components=2, random_state=42,
                                    init=pca_init, perplexity=30, n_iter=750,
                                    method="barnes_hut",
                                ).fit_transform(embeddings)

                        else:  # PCA
                            from sklearn.decomposition import PCA as _PCA
                            _pca_model = _PCA(n_components=2).fit(embeddings)
                            coords = _pca_model.transform(embeddings)
                            _pca_var = _pca_model.explained_variance_ratio_ * 100

                        # Store coords, the method that was *actually* run (not
                        # just selected), and PCA variance if available.
                        st.session_state[proj_key] = coords
                        st.session_state[proj_method_key] = _actual_method
                        st.session_state["_proj_pca_var"] = _pca_var

                    except ImportError as e:
                        missing = str(e).split("'")[1] if "'" in str(e) else str(e)
                        st.error(
                            f"{proj_method} requires the `{missing}` package. "
                            f"Install it with: `pip install {missing}` — "
                            f"falling back to PCA."
                        )
                        from sklearn.decomposition import PCA as _PCA
                        _pca_model = _PCA(n_components=2).fit(embeddings)
                        coords = _pca_model.transform(embeddings)
                        st.session_state[proj_key] = coords
                        st.session_state[proj_method_key] = "PCA"
                        st.session_state["_proj_pca_var"] = _pca_model.explained_variance_ratio_ * 100

            coords = st.session_state[proj_key]

            # ── Build plot dataframe ──────────────────────────────────────────
            plot_df = df.copy().reset_index(drop=True)
            plot_df["x"] = coords[:, 0]
            plot_df["y"] = coords[:, 1]
            plot_df["is_noise"] = plot_df["cluster_id"] == -1
            plot_df["label"] = plot_df["cluster_id"].apply(
                lambda c: "noise" if c == -1 else f"cluster {int(c)}"
            )
            # Truncate long signatures for hover
            plot_df["sig_short"] = plot_df["signature"].str[:60]
            plot_df["cluster_str"] = plot_df["cluster_id"].astype(str)
            plot_df["severity_str"] = plot_df["severity"].fillna("?").astype(str)

            if not show_noise_map:
                plot_df = plot_df[~plot_df["is_noise"]].reset_index(drop=True)

            if len(plot_df) == 0:
                st.info("No points to display — enable noise or run with lower min_samples.")
            else:
                # ── Colour mapping ────────────────────────────────────────────
                NOISE_COLOR = "#1e2a32"

                # 20 maximally distinct hues — chosen by hand to maximise perceptual
                # distance between every adjacent pair. No two neighbours share the
                # same hue family. Verified against deuteranopia/protanopia simulations.
                PALETTE = [
                    "#E63946",  # 0  vivid red
                    "#2EC4B6",  # 1  teal       — max dist from red
                    "#F4A261",  # 2  warm orange
                    "#457B9D",  # 3  steel blue  — max dist from orange
                    "#A8DADC",  # 4  ice blue
                    "#C77DFF",  # 5  violet
                    "#F9C74F",  # 6  golden yellow
                    "#06D6A0",  # 7  emerald
                    "#FF6B6B",  # 8  coral        — diff lightness from red
                    "#118AB2",  # 9  ocean blue
                    "#FFB703",  # 10 amber
                    "#8338EC",  # 11 deep purple   — max dist from amber
                    "#43AA8B",  # 12 sage green
                    "#F72585",  # 13 hot pink      — max dist from green
                    "#90E0EF",  # 14 powder blue
                    "#FB5607",  # 15 burnt orange
                    "#B5E48C",  # 16 lime          — light, distinct from emerald
                    "#6A0572",  # 17 dark magenta  — max dist from lime
                    "#FFD166",  # 18 light yellow  — diff lightness from amber
                    "#264653",  # 19 dark teal     — max dist from light yellow
                ]

                # For >20 clusters generate additional colours via golden-angle HSL
                # stepping so colours never repeat exactly, just cycle with lightness shift.
                def _get_color(idx: int) -> str:
                    if idx < len(PALETTE):
                        return PALETTE[idx]
                    import colorsys
                    golden = 0.618033988749895
                    hue = (idx * golden) % 1.0
                    lightness = 0.55 if idx % 2 == 0 else 0.70
                    r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.85)
                    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

                unique_clusters = sorted(
                    [c for c in plot_df["cluster_id"].unique() if c != -1]
                )

                # ── Build per-colour-mode trace groups ────────────────────────
                # Each mode defines: groups to iterate, label per group,
                # colour per group, and subset of clustered_df.
                # This fixes the bug where color_discrete_map was computed
                # but never applied to the actual Scattergl traces.

                if color_by == "severity":
                    SEV_COLORS = {
                        "1": "#E63946",  # red    — critical
                        "2": "#F4A261",  # orange — high
                        "3": "#2EC4B6",  # teal   — medium (not green to avoid "safe" signal)
                        "4": "#A8DADC",  # ice    — low/info
                        "?": "#3a4a50",  # dim    — unknown
                    }
                    plot_df["_sev_key"] = plot_df["severity"].fillna("?").astype(str)
                    trace_groups = [
                        (key, f"Severity {key}", SEV_COLORS.get(key, "#888888"),
                         plot_df[(~plot_df["is_noise"]) & (plot_df["_sev_key"] == key)])
                        for key in ["1", "2", "3", "4", "?"]
                        if (plot_df["_sev_key"] == key).any()
                    ]

                elif color_by == "proto":
                    plot_df["_proto_key"] = plot_df["proto"].fillna("unknown").str.upper()
                    unique_protos = sorted(plot_df["_proto_key"].unique())
                    trace_groups = [
                        (proto, proto, _get_color(i),
                         plot_df[(~plot_df["is_noise"]) & (plot_df["_proto_key"] == proto)])
                        for i, proto in enumerate(unique_protos)
                    ]

                elif color_by == "category":
                    plot_df["_cat_key"] = plot_df["category"].fillna("(none)").astype(str)
                    unique_cats = sorted(plot_df["_cat_key"].unique())
                    trace_groups = [
                        (cat, cat, _get_color(i),
                         plot_df[(~plot_df["is_noise"]) & (plot_df["_cat_key"] == cat)])
                        for i, cat in enumerate(unique_cats)
                    ]

                elif color_by == "src_ip":
                    # Top 15 source IPs get their own colour; the rest become "other"
                    top_ips = (
                        plot_df[~plot_df["is_noise"]]["src_ip"]
                        .value_counts()
                        .head(15)
                        .index.tolist()
                    )
                    plot_df["_ip_key"] = plot_df["src_ip"].apply(
                        lambda ip: ip if ip in top_ips else "other"
                    )
                    ordered = top_ips + (["other"] if "other" in plot_df["_ip_key"].values else [])
                    trace_groups = [
                        (ip, ip, _get_color(i) if ip != "other" else "#2a3a44",
                         plot_df[(~plot_df["is_noise"]) & (plot_df["_ip_key"] == ip)])
                        for i, ip in enumerate(ordered)
                        if len(plot_df[(~plot_df["is_noise"]) & (plot_df["_ip_key"] == ip)]) > 0
                    ]

                else:  # cluster_id (default)
                    # Label by cluster number, colour by cluster index.
                    # incident_id mode is separate — do NOT use incident_id here
                    # or both views look identical.
                    trace_groups = [
                        (
                            f"CL-{cid}",            # key
                            f"Cluster {cid}",        # legend label — cluster number, not incident
                            _get_color(i),
                            plot_df[(~plot_df["is_noise"]) & (plot_df["cluster_id"] == cid)],
                        )
                        for i, cid in enumerate(unique_clusters)
                    ]

                # ── Axis labels & descriptions ────────────────────────────────
                # Use the method that was *actually* run (stored at projection time),
                # not the radio value — they differ when t-SNE fell back to PCA.
                _active_method = st.session_state.get(proj_method_key, proj_method)
                _pca_var = st.session_state.get("_proj_pca_var")

                if _active_method == "PCA":
                    if _pca_var is not None:
                        x_label = (
                            f"PC1  ·  {_pca_var[0]:.1f}% of variance explained  "
                            f"— alerts spread horizontally by their biggest overall difference"
                        )
                        y_label = (
                            f"PC2  ·  {_pca_var[1]:.1f}% of variance explained  "
                            f"— second biggest difference runs vertically"
                        )
                    else:
                        x_label = "PC1  —  primary axis of embedding variance"
                        y_label = "PC2  —  secondary axis of embedding variance"
                    axis_ticks_meaningful = True

                elif _active_method == "UMAP":
                    x_label = (
                        "UMAP-1  —  arbitrary scale; "
                        "dots close together = semantically similar alerts"
                    )
                    y_label = (
                        "UMAP-2  —  arbitrary scale; "
                        "dots close together = semantically similar alerts"
                    )
                    axis_ticks_meaningful = False

                else:  # t-SNE
                    x_label = (
                        "t-SNE-1  —  arbitrary scale; "
                        "focus on cluster shape and separation, not axis values"
                    )
                    y_label = (
                        "t-SNE-2  —  arbitrary scale; "
                        "focus on cluster shape and separation, not axis values"
                    )
                    axis_ticks_meaningful = False

                # ── Scatter plot ──────────────────────────────────────────────
                # Split noise and clustered for layering (noise underneath)
                noise_df     = plot_df[plot_df["is_noise"]]
                clustered_df = plot_df[~plot_df["is_noise"]]

                fig = go.Figure()

                # Noise layer
                if show_noise_map and len(noise_df) > 0:
                    fig.add_trace(go.Scattergl(
                        x=noise_df["x"], y=noise_df["y"],
                        mode="markers",
                        name="noise",
                        marker=dict(
                            color=NOISE_COLOR,
                            size=4,
                            opacity=0.35,
                            line=dict(width=0),
                        ),
                        hovertemplate=(
                            "<b>NOISE</b><br>"
                            "src: %{customdata[0]}<br>"
                            "sig: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=noise_df[["src_ip", "sig_short"]].values,
                    ))

                # Clustered layer — one trace per group so the legend reflects
                # the chosen colour_by attribute and colours are actually applied.
                for _key, _label, _color, sub in trace_groups:
                    if len(sub) == 0:
                        continue
                    fig.add_trace(go.Scattergl(
                        x=sub["x"], y=sub["y"],
                        mode="markers",
                        name=str(_label),
                        marker=dict(
                            color=_color,
                            size=7,
                            opacity=0.88,
                            line=dict(width=0.6, color="rgba(0,0,0,0.25)"),
                        ),
                        hovertemplate=(
                            f"<b>{_label}</b><br>"
                            "src: %{customdata[0]} → dst: %{customdata[1]}<br>"
                            "sig: %{customdata[2]}<br>"
                            "sev: %{customdata[3]}  proto: %{customdata[4]}<br>"
                            "<extra></extra>"
                        ),
                        customdata=sub[["src_ip", "dest_ip", "sig_short", "severity_str", "proto"]].values,
                    ))

                fig.update_layout(
                    paper_bgcolor="#0a0c10",
                    plot_bgcolor="#080b0f",
                    font=dict(family="IBM Plex Mono", color="#4a6070", size=11),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis=dict(
                        title=dict(
                            text=x_label,
                            font=dict(size=11, color="#4a8090", family="IBM Plex Mono"),
                        ),
                        gridcolor="#111820",
                        zeroline=True,
                        zerolinecolor="#1a2530",
                        zerolinewidth=1,
                        showticklabels=axis_ticks_meaningful,
                        tickfont=dict(size=9, color="#2a3a4a", family="IBM Plex Mono"),
                        tickformat=".2f",
                    ),
                    yaxis=dict(
                        title=dict(
                            text=y_label,
                            font=dict(size=11, color="#4a8090", family="IBM Plex Mono"),
                        ),
                        gridcolor="#111820",
                        zeroline=True,
                        zerolinecolor="#1a2530",
                        zerolinewidth=1,
                        showticklabels=axis_ticks_meaningful,
                        tickfont=dict(size=9, color="#2a3a4a", family="IBM Plex Mono"),
                        tickformat=".2f",
                    ),
                    legend=dict(
                        bgcolor="#0f1318",
                        bordercolor="#1e2530",
                        borderwidth=1,
                        font=dict(size=10, color="#7a8799"),
                        itemsizing="constant",
                        title=dict(text="INCIDENT / CLUSTER", font=dict(size=10, color="#3d5066")),
                    ),
                    title=dict(
                        text=(
                            f"{st.session_state.get(proj_method_key, proj_method)} projection  ·  "
                            f"{len(clustered_df):,} clustered  ·  "
                            f"{len(noise_df):,} noise  ·  "
                            f"{len(unique_clusters)} clusters"
                        ),
                        font=dict(size=12, color="#3d5066", family="IBM Plex Mono"),
                        x=0.01,
                    ),
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)

                # ── Description panel ────────────────────────────────────────
                _active_method = st.session_state.get(proj_method_key, proj_method)
                _method_explanations = {
                    "PCA": (
                        "PCA (Principal Component Analysis) rotates the 384-dimensional "
                        "embedding space so the axis with the most variation across all "
                        "alerts becomes the horizontal axis (PC1), and the next most "
                        "varied becomes vertical (PC2). "
                        "<b>The tick numbers are real and comparable</b> — a dot at x=2.5 "
                        "is genuinely further along that axis than one at x=0.8. "
                        "Tight groups of same-colour dots mean those alerts have very "
                        "similar signatures, ports, and protocols."
                    ),
                    "UMAP": (
                        "UMAP (Uniform Manifold Approximation and Projection) learns the "
                        "local neighbourhood structure of the 384-dimensional embedding "
                        "space and flattens it to 2D while preserving which alerts are "
                        "close to which. "
                        "<b>The axis numbers are arbitrary</b> — ignore them. "
                        "What matters: dots that appear close together are semantically "
                        "similar alerts. A tight island of colour = a strong, coherent "
                        "cluster. A blob spread across the map = a loose grouping. "
                        "UMAP gives the most faithful picture of true cluster separation."
                    ),
                    "t-SNE": (
                        "t-SNE (t-distributed Stochastic Neighbour Embedding) pulls "
                        "similar alerts together and pushes dissimilar ones apart, "
                        "emphasising local neighbourhood structure. "
                        "<b>The axis numbers are arbitrary</b> — ignore them. "
                        "Cluster size and inter-cluster distance on this plot are "
                        "not proportional to real similarity differences; use it to "
                        "spot whether clusters are clean and tight, not to measure "
                        "how far apart two incidents are."
                    ),
                }
                _method_desc = _method_explanations.get(_active_method, "")

                st.markdown(
                    f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px;">

                      <div style="background:#0f1318;border:1px solid #1a2530;border-top:2px solid #00d4ff;
                                  padding:12px 14px;border-radius:2px;">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                    color:#3d5066;letter-spacing:0.12em;text-transform:uppercase;
                                    margin-bottom:6px;">How to read this chart</div>
                        <div style="font-size:0.8rem;color:#7a8da0;line-height:1.55;">
                          <b style="color:#c8d0db;">Each dot</b> = one alert from your eve.json.<br>
                          <b style="color:#c8d0db;">Dot colour</b> = the cluster or attribute selected in "Colour by".<br>
                          <b style="color:#c8d0db;">Proximity</b> = similarity — dots near each other fired for the
                          same kind of activity (same signature family, port pattern, or protocol).<br>
                          <b style="color:#2a3a4a;">Grey dots</b> = noise alerts the algorithm couldn't fit into any cluster.
                        </div>
                      </div>

                      <div style="background:#0f1318;border:1px solid #1a2530;border-top:2px solid #FFB800;
                                  padding:12px 14px;border-radius:2px;">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                    color:#3d5066;letter-spacing:0.12em;text-transform:uppercase;
                                    margin-bottom:6px;">What the axes mean — {_active_method}</div>
                        <div style="font-size:0.8rem;color:#7a8da0;line-height:1.55;">
                          {_method_desc}
                        </div>
                      </div>

                      <div style="background:#0f1318;border:1px solid #1a2530;border-top:2px solid #00e87a;
                                  padding:12px 14px;border-radius:2px;">
                        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                    color:#3d5066;letter-spacing:0.12em;text-transform:uppercase;
                                    margin-bottom:6px;">Patterns to look for</div>
                        <div style="font-size:0.8rem;color:#7a8da0;line-height:1.55;">
                          <b style="color:#c8d0db;">Tight island</b> = strong, coherent incident (good cluster).<br>
                          <b style="color:#c8d0db;">Elongated streak</b> = alerts from the same src_ip over time,
                          with slight signature variation.<br>
                          <b style="color:#c8d0db;">Two colours mixed</b> = possible cluster over-merge — consider
                          lowering ε or xi.<br>
                          <b style="color:#c8d0db;">Scattered noise around a cluster</b> = related activity the
                          algorithm was uncertain about — try increasing the time window.
                        </div>
                      </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Distributions tab ─────────────────────────────────────────────────────
    with tab_dist:
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            has_plotly = True
        except ImportError:
            has_plotly = False

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">Alerts per Cluster</div>', unsafe_allow_html=True)
            cluster_counts = (
                df[df["cluster_id"] != -1]["cluster_id"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            cluster_counts.columns = ["cluster_id", "count"]
            if has_plotly:
                fig = px.bar(
                    cluster_counts.head(40),
                    x="cluster_id", y="count",
                    template="plotly_dark",
                    color_discrete_sequence=["#2196F3"],
                )
                fig.update_layout(
                    paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
                    font=dict(family="IBM Plex Mono", color="#4a6070", size=11),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor="#1a2028"),
                    yaxis=dict(gridcolor="#1a2028"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(cluster_counts.set_index("cluster_id")["count"].head(40))

        with col_b:
            st.markdown('<div class="section-label">Top Signatures</div>', unsafe_allow_html=True)
            sig_counts = df["signature"].value_counts().head(15).reset_index()
            sig_counts.columns = ["signature", "count"]
            sig_counts["signature_short"] = sig_counts["signature"].str[:40]
            if has_plotly:
                fig2 = px.bar(
                    sig_counts,
                    x="count", y="signature_short",
                    orientation="h",
                    template="plotly_dark",
                    color_discrete_sequence=["#FFB800"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
                    font=dict(family="IBM Plex Mono", color="#4a6070", size=10),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor="#1a2028"),
                    yaxis=dict(gridcolor="#1a2028"),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.bar_chart(sig_counts.set_index("signature_short")["count"])

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="section-label">Severity Distribution</div>', unsafe_allow_html=True)
            sev_counts = df["severity"].value_counts().sort_index().reset_index()
            sev_counts.columns = ["severity", "count"]
            sev_counts["severity"] = sev_counts["severity"].astype(str)
            if has_plotly:
                fig3 = px.pie(
                    sev_counts, names="severity", values="count",
                    template="plotly_dark",
                    color_discrete_sequence=["#FF2D55", "#FFB800", "#30D158", "#64D2FF"],
                )
                fig3.update_layout(
                    paper_bgcolor="#0a0c10",
                    font=dict(family="IBM Plex Mono", color="#4a6070", size=11),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.bar_chart(sev_counts.set_index("severity")["count"])

        with col_d:
            st.markdown('<div class="section-label">Top Source IPs</div>', unsafe_allow_html=True)
            ip_counts = df["src_ip"].value_counts().head(15).reset_index()
            ip_counts.columns = ["src_ip", "count"]
            if has_plotly:
                fig4 = px.bar(
                    ip_counts, x="count", y="src_ip",
                    orientation="h",
                    template="plotly_dark",
                    color_discrete_sequence=["#30D158"],
                )
                fig4.update_layout(
                    paper_bgcolor="#0a0c10", plot_bgcolor="#0a0c10",
                    font=dict(family="IBM Plex Mono", color="#4a6070", size=10),
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(gridcolor="#1a2028"),
                    yaxis=dict(gridcolor="#1a2028"),
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.bar_chart(ip_counts.set_index("src_ip")["count"])

    # ── Download tab ──────────────────────────────────────────────────────────
    with tab_download:
        st.markdown('<div class="section-label">Export Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            parquet_path = r.get("parquet_path")
            if parquet_path and os.path.exists(parquet_path):
                with open(parquet_path, "rb") as pf:
                    st.download_button(
                        "⬇ alerts_clustered.parquet",
                        data=pf,
                        file_name="alerts_clustered.parquet",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
            else:
                st.button("⬇ alerts_clustered.parquet", disabled=True, use_container_width=True)
            st.caption("Full clustered alert table")

        with col2:
            st.download_button(
                "⬇ incidents.json",
                data=r["incidents_json"],
                file_name="incidents.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Incident summaries (JSON)")

        with col3:
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                "⬇ run_stats.json",
                data=stats_json,
                file_name="run_stats.json",
                mime="application/json",
                use_container_width=True,
            )
            st.caption("Pipeline run statistics")

        st.markdown('<div class="section-label">Raw Run Stats</div>', unsafe_allow_html=True)
        st.code(json.dumps(stats, indent=2), language="json")