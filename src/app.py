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
    .stat-card.orange { border-top-color: #ff7c3a; }
    .stat-card.red    { border-top-color: #ff3a5c; }
    .stat-card.green  { border-top-color: #00e87a; }
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
    .incident-card.sev-high   { border-left-color: #ff3a5c; }
    .incident-card.sev-medium { border-left-color: #ff7c3a; }
    .incident-card.sev-low    { border-left-color: #00d4ff; }
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
    .inc-badge.red   { background: #2a1520; color: #ff5570; }
    .inc-badge.orange{ background: #2a1a10; color: #ff7c3a; }

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
st.markdown('<div class="main-subheader">Semantic grouping & incident synthesis from eve.json streams</div>', unsafe_allow_html=True)

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

    algo = st.radio("Algorithm", ["dbscan", "hierarchical"], index=0,
        help="DBSCAN: density-based. Hierarchical: agglomerative average-link.")

    if algo == "dbscan":
        eps = st.slider("DBSCAN ε (cosine dist)", 0.05, 0.80, 0.25, 0.01,
            help="Max cosine distance between neighbours.")
        dist_threshold = 0.25
    else:
        dist_threshold = st.slider("Distance threshold (cosine)", 0.05, 0.80, 0.25, 0.01,
            help="Agglomerative clustering cut threshold.")
        eps = 0.25

    st.markdown("---")
    st.markdown("### 🤖 Embedding Model")

    model_name = st.selectbox(
        "Sentence Transformer",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
        ],
        index=0,
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
    if sev <= 1:
        return "sev-high"
    if sev <= 2:
        return "sev-medium"
    return "sev-low"


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
    sev_badge = (
        f'<span class="inc-badge red">SEV {sev}</span>' if sev and sev <= 1 else
        f'<span class="inc-badge orange">SEV {sev}</span>' if sev and sev <= 2 else
        f'<span class="inc-badge">SEV {sev}</span>' if sev else ""
    )
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

    # Stream to disk in 64 MB chunks — handles files up to 1 GB without
    # loading the entire buffer into Python memory at once.
    CHUNK = 64 * 1024 * 1024  # 64 MB
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
    tab_inc, tab_alerts, tab_dist, tab_download = st.tabs([
        "Incidents", "Alert Table", "Distributions", "Download"
    ])

    # ── Incidents tab ─────────────────────────────────────────────────────────
    with tab_inc:
        if not incidents:
            st.info("No incidents formed. Try lowering min_samples or increasing the time window.")
        else:
            # Filter controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search = st.text_input("Search incidents", placeholder="signature, IP…", label_visibility="collapsed")
            with col2:
                sev_filter = st.selectbox("Max severity", ["All", "1 (High)", "2 (Med)", "3 (Low)"], label_visibility="collapsed")
            with col3:
                sort_by = st.selectbox("Sort by", ["Time (asc)", "Alert count (desc)", "Severity (asc)"], label_visibility="collapsed")

            filtered = incidents.copy()

            if search:
                s = search.lower()
                filtered = [
                    i for i in filtered
                    if s in i["summary"].lower()
                    or any(s in sig.lower() for sig in i["top_signatures"])
                    or any(s in ip for ip in i["src_ips"] + i["dest_ips"])
                ]
            if sev_filter != "All":
                max_s = int(sev_filter[0])
                filtered = [i for i in filtered if (i.get("max_severity") or 99) <= max_s]
            if sort_by == "Alert count (desc)":
                filtered = sorted(filtered, key=lambda x: -x["alert_count"])
            elif sort_by == "Severity (asc)":
                filtered = sorted(filtered, key=lambda x: (x.get("max_severity") or 99))

            st.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d5066;margin-bottom:12px;">'
                f'SHOWING {len(filtered)} / {len(incidents)} INCIDENTS</div>',
                unsafe_allow_html=True,
            )

            for inc in filtered:
                render_incident_card(inc)
                with st.expander("Representative alerts"):
                    rep = pd.DataFrame(inc["representative_alerts"])
                    st.dataframe(rep, use_container_width=True, hide_index=True)

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
            mask = disp.astype(str).apply(lambda row: q.lower() in row.str.lower().str.cat(sep=" "), axis=1)
            disp = disp[mask]

        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d5066;margin-bottom:8px;">'
            f'{len(disp):,} alerts displayed</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(disp.head(2000), use_container_width=True, hide_index=True, height=450)
        if len(disp) > 2000:
            st.caption(f"Displaying first 2,000 of {len(disp):,}. Download the full parquet for all rows.")

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
                    color_discrete_sequence=["#00d4ff"],
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
                    color_discrete_sequence=["#ff7c3a"],
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
                    color_discrete_sequence=["#ff3a5c", "#ff7c3a", "#00d4ff", "#00e87a"],
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
                    color_discrete_sequence=["#00e87a"],
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