"""
Cluster Map figure for the Streamlit GUI (app.py).

Kept free of Streamlit so the figure can be built and checked on its own.

What each visual channel means (the same on every run):
  position = 2-D projection of the alert embedding, i.e. what the alert LOOKS LIKE
             (signature, category, severity, protocol, destination port, internal/external).
             It does NOT encode source IP or time — those only decide incident grouping.
             In the default incident view every incident is ONE bubble at the centre of
             its alerts, nudged just far enough that no two bubbles cover each other.
  size     = how many alerts the bubble holds (incident view) or how many
             identical-looking alerts are stacked on that exact spot (alert view).
  colour   = the grouping picked in "Colour by". Capped at 3 colours + grey so every
             colour stays tellable apart on a scatter, including for colour-blind analysts.
  ring     = unclustered (noise) alert that belongs to no incident.
"""

import html
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Plain-English names for the families assigned in incidents.normalize_family().
# The "(unclassified)" ones are port-based fallbacks: no attack rule matched.
FAMILY_LABELS = {
    "malware_c2": "Malware C2 / beaconing",
    "smb_worm": "SMB exploit / worm",
    "java_rce": "Java RCE (Log4j)",
    "web_sqli": "SQL injection",
    "web_exploitation": "Web exploitation",
    "ssh_bruteforce": "SSH brute force",
    "rdp_bruteforce": "RDP brute force",
    "recon_scan": "Recon / scanning",
    "dns_suspicious": "Suspicious DNS",
    "ssh_activity": "SSH traffic (unclassified)",
    "rdp_activity": "RDP traffic (unclassified)",
    "smb_activity": "SMB traffic (unclassified)",
    "dns_activity": "DNS traffic (unclassified)",
    "web_activity": "Web traffic (unclassified)",
    "misc_activity": "Unclassified activity",
}

SEVERITY_NAMES = {1: "High", 2: "Medium", 3: "Low"}

# Validated with the dataviz palette checker against the plot surface below
# (all-pairs, dark mode): blue / orange / aqua clear the CVD and normal-vision floors.
HIGHLIGHT_COLORS = ["#3987e5", "#d95926", "#199e70"]
OTHER_COLOR = "#5a6470"     # every non-highlighted group; 3.3:1 on the surface
NOISE_COLOR = "#7a8799"     # drawn as a hollow ring, so it differs by shape too
# Severity: status red / orange for the two levels that need attention, neutral slate for low.
SEVERITY_COLORS = {1: "#d03b3b", 2: "#ec835a", 3: "#8a9db0"}

SURFACE = "#080b0f"
INK = "#c8d0db"
MUTED_INK = "#7a8799"

COLOR_BY_OPTIONS = ["Incident", "Severity", "Protocol", "Category", "Source IP"]

_OTHER = "__other__"


def family_label(family: Optional[str]) -> str:
    return FAMILY_LABELS.get(family or "", family or "Unclassified activity")


def severity_label(sev) -> str:
    if sev is None or pd.isna(sev):
        return "Sev ? · Unknown"
    return f"Sev {int(sev)} · {SEVERITY_NAMES.get(int(sev), 'Unknown')}"


def incident_option(inc: dict) -> str:
    """Label used in the highlight picker, e.g. 'INC-00004 · Recon / scanning · risk 68 · 810 alerts'."""
    return (
        f"{inc['incident_id']} · {family_label(inc.get('family'))} · "
        f"risk {inc.get('risk_score', '?')} · {inc['alert_count']:,} alerts"
    )


def top_incidents(incidents: List[dict], n: int = len(HIGHLIGHT_COLORS)) -> List[str]:
    ranked = sorted(incidents, key=lambda i: (-i.get("risk_score", 0), -i["alert_count"]))
    return [i["incident_id"] for i in ranked[:n]]


def assign_slots(selected: Iterable[str], previous: Dict[str, int]) -> Dict[str, int]:
    """Give each highlighted incident a colour slot, keeping the slot it already had.

    Colour follows the incident, not its position in the picker, so removing one
    highlight never repaints the others.
    """
    selected = list(selected)
    slots = {i: s for i, s in previous.items() if i in selected}
    free = [s for s in range(len(HIGHLIGHT_COLORS)) if s not in slots.values()]
    for inc_id in selected:
        if inc_id not in slots and free:
            slots[inc_id] = free.pop(0)
    return slots


def incident_table(incidents: List[dict]) -> pd.DataFrame:
    """Table twin of the map: every incident, highest risk first."""
    rows = []
    for inc in sorted(incidents, key=lambda i: (-i.get("risk_score", 0), -i["alert_count"])):
        rows.append({
            "Incident": inc["incident_id"],
            "Attack type": family_label(inc.get("family")),
            "Risk": inc.get("risk_score"),
            "Severity": severity_label(inc.get("max_severity")),
            "Alerts": inc["alert_count"],
            "Source IPs": ", ".join(inc["src_ips"][:3]) + (" …" if len(inc["src_ips"]) > 3 else ""),
            "Targets": len(inc["dest_ips"]),
            "Ports": ", ".join(map(str, inc["dest_ports"][:6])) + (" …" if len(inc["dest_ports"]) > 6 else ""),
            "Start": inc["start_time"][:19].replace("T", " "),
            "End": inc["end_time"][:19].replace("T", " "),
            "Summary": inc["summary"],
        })
    return pd.DataFrame(rows)


# ── Internals ─────────────────────────────────────────────────────────────────

def _few(values: pd.Series, k: int = 2) -> str:
    """'a, b (+3 more)' — most frequent values first."""
    vc = values.dropna().astype(str).value_counts()
    if vc.empty:
        return "—"
    shown = ", ".join(vc.index[:k])
    return shown + (f" (+{len(vc) - k} more)" if len(vc) > k else "")


def _mode(values: pd.Series):
    vc = values.value_counts()
    return vc.index[0] if len(vc) else None


def _stack(sub: pd.DataFrame) -> pd.DataFrame:
    """Collapse alerts sharing a ~2px spot (same `_bin`) into one marker with a count.

    Identical alerts embed to the identical point, and near-identical ones (e.g. only the
    source-port bucket differs) land within a pixel of each other; drawing them as separate
    markers would hide their volume, so marker size shows the total instead.
    """
    g = sub.groupby(["_bx", "_by"], sort=False)
    out = g.agg(
        x=("x", "mean"),
        y=("y", "mean"),
        n=("x", "size"),
        n_sigs=("signature", "nunique"),
        severity=("severity", "min"),     # Suricata: lower = more severe
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    )
    for col in ("signature", "category", "proto", "dest_port"):
        out[col] = g[col].agg(_mode)
    out["incidents"] = g["incident_id"].agg(lambda s: _few(s, 2))
    out["incident_ids"] = g["incident_id"].agg(lambda s: list(s.dropna().unique()))
    out["src"] = g["src_ip"].agg(_few)
    out["dst"] = g["dest_ip"].agg(_few)
    # Biggest first so small markers are drawn on top and stay hoverable.
    return out.reset_index(drop=True).sort_values("n", ascending=False)


def _hover(stack: pd.DataFrame, inc_by_id: Dict[str, dict], noise: bool) -> List[str]:
    e = html.escape
    texts = []
    for r in stack.itertuples(index=False):
        if noise:
            head = "<b>Unclustered alert</b> — not part of any incident"
        elif len(r.incident_ids) == 1 and r.incident_ids[0] in inc_by_id:
            inc = inc_by_id[r.incident_ids[0]]
            head = (
                f"<b>{e(inc['incident_id'])}</b> · {e(family_label(inc.get('family')))}"
                f" · risk {inc.get('risk_score', '?')}"
            )
        else:
            head = f"<b>{len(r.incident_ids)} incidents here:</b> {e(r.incidents)}"

        what = "alert" if r.n == 1 else "alerts"
        port = "" if pd.isna(r.dest_port) else f" → port {int(r.dest_port)}"
        category = "—" if pd.isna(r.category) else str(r.category)
        proto = "?" if pd.isna(r.proto) else str(r.proto)
        more_sigs = f"  (+{r.n_sigs - 1} other signatures)" if r.n_sigs > 1 else ""
        texts.append(
            f"{head}<br>"
            f"{r.n:,} {what} at this spot<br>"
            f"{e(str(r.signature)[:80])}{more_sigs}<br>"
            f"{e(category)} · {severity_label(r.severity)} · {e(proto)}{port}<br>"
            f"{e(r.src)} → {e(r.dst)}<br>"
            f"{e(str(r.first_seen)[:16].replace('T', ' '))} → {e(str(r.last_seen)[:16].replace('T', ' '))}"
        )
    return texts


def _top_values(clustered: pd.DataFrame, col: str, n: int) -> List[str]:
    return clustered[col].value_counts().head(n).index.tolist()


def _groups(pts: pd.DataFrame, incidents: List[dict], color_by: str, slots: Dict[str, int]):
    """Return [(legend name, colour, alerts)] for clustered alerts, grey group first."""
    clustered = pts[~pts["is_noise"]]
    n_col = len(HIGHLIGHT_COLORS)

    if color_by == "Severity":
        sev = clustered["severity"]
        groups = []
        for level in (1, 2, 3):
            sub = clustered[sev == level]
            if len(sub):
                groups.append((f"{severity_label(level)} ({len(sub):,})", SEVERITY_COLORS[level], sub))
        rest = clustered[~sev.isin([1, 2, 3])]
        if len(rest):
            groups.insert(0, (f"Sev 4 / unknown ({len(rest):,})", OTHER_COLOR, rest))
        return groups

    if color_by == "Incident":
        by_id = {i["incident_id"]: i for i in incidents}
        hl = [i for i in sorted(slots, key=slots.get) if i in by_id]
        rest = clustered[~clustered["incident_id"].isin(hl)]
        n_other = rest["incident_id"].nunique()
        groups = [(f"Other incidents ({n_other})", OTHER_COLOR, rest)] if len(rest) else []
        for inc_id in hl:
            inc = by_id[inc_id]
            groups.append((
                f"{inc_id} · {family_label(inc.get('family'))} ({inc['alert_count']:,})",
                HIGHLIGHT_COLORS[slots[inc_id]],
                clustered[clustered["incident_id"] == inc_id],
            ))
        return groups

    # Protocol / Category / Source IP: the 3 biggest values get colours, the rest fold into grey.
    col = {"Protocol": "_proto", "Category": "_category", "Source IP": "src_ip"}[color_by]
    top = _top_values(clustered, col, n_col)
    rest = clustered[~clustered[col].isin(top)]
    groups = [(f"Other ({rest[col].nunique()})", OTHER_COLOR, rest)] if len(rest) else []
    for i, val in enumerate(top):
        sub = clustered[clustered[col] == val]
        groups.append((f"{val} ({len(sub):,})", HIGHLIGHT_COLORS[i], sub))
    return groups


_COLOR_MEANING = {
    "Incident": "the highlighted incidents (all others grey)",
    "Severity": "worst Suricata severity of the alert",
    "Protocol": "protocol (top 3, rest grey)",
    "Category": "Suricata alert category (top 3, rest grey)",
    "Source IP": "source IP (top 3, rest grey)",
}


def _sizes(counts, max_count) -> np.ndarray:
    # Marker area grows with the number of alerts (diameter ~ sqrt(count)).
    return 6 + 22 * np.sqrt(np.asarray(counts) / max_count)


def _one_bubble_per_incident(pts: pd.DataFrame) -> None:
    """Move each clustered alert onto its incident's centre, then push overlapping bubbles apart.

    Plotting alerts individually gives a scan one dot per port tried and a 600-alert brute
    force a single dot, and stacks incidents that look alike on the same spot. One bubble
    per incident shows what the clustering actually produced.
    """
    clustered = ~pts["is_noise"]
    inc = pts[clustered].groupby("incident_id").agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size"))
    if len(inc) < 2:
        return
    diam = _sizes(inc["n"], inc["n"].max())
    lo = pts[["x", "y"]].min().to_numpy()
    span = (pts[["x", "y"]].max().to_numpy() - lo).clip(1e-9)
    box = np.array([1000.0, 520.0])                      # approx. plot area in px
    p = (inc[["x", "y"]].to_numpy() - lo) / span * box
    # Incidents with identical alerts start on the same point; a tiny spiral gives them a direction.
    k = np.arange(len(p))
    p += 0.5 * np.sqrt(k)[:, None] * np.column_stack([np.cos(2.4 * k), np.sin(2.4 * k)])
    need = (diam[:, None] + diam[None, :]) / 2 + 4       # centre distance at which two bubbles clear
    for _ in range(300):
        d = p[:, None, :] - p[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(dist, np.inf)
        push = np.clip(need - dist, 0, None)
        if not push.any():
            break
        p += 0.3 * (d / dist[..., None] * push[..., None]).sum(1)
    inc[["x", "y"]] = p / box * span + lo
    pts.loc[clustered, "x"] = pts.loc[clustered, "incident_id"].map(inc["x"])
    pts.loc[clustered, "y"] = pts.loc[clustered, "incident_id"].map(inc["y"])


# ── Public entry point ────────────────────────────────────────────────────────

def build_cluster_map(
    df: pd.DataFrame,
    coords: np.ndarray,
    incidents: List[dict],
    color_by: str = "Incident",
    slots: Optional[Dict[str, int]] = None,
    show_noise: bool = True,
    method: str = "UMAP",
    x_title: str = "",
    y_title: str = "",
    show_ticks: bool = False,
    by_incident: bool = True,
) -> go.Figure:
    """Build the Cluster Map. `slots` maps highlighted incident_id -> colour slot (see assign_slots).

    by_incident=True draws one bubble per incident; False draws every distinct alert.
    """
    slots = slots or {}
    inc_by_id = {i["incident_id"]: i for i in incidents}

    pts = df.reset_index(drop=True).copy()
    pts["x"] = np.asarray(coords[:, 0], dtype=float)   # projections come back as float32
    pts["y"] = np.asarray(coords[:, 1], dtype=float)
    pts["is_noise"] = pts["cluster_id"] == -1
    pts["_proto"] = pts["proto"].fillna("unknown").astype(str).str.upper()
    pts["_category"] = pts["category"].fillna("(none)").astype(str)
    if by_incident:
        _one_bubble_per_incident(pts)
    # Snap to a 300x300 grid (~2px on screen) so visually coincident alerts stack into one marker.
    for axis_col, bin_col in (("x", "_bx"), ("y", "_by")):
        lo, span = pts[axis_col].min(), pts[axis_col].max() - pts[axis_col].min()
        pts[bin_col] = np.round((pts[axis_col] - lo) / (span / 300 if span > 0 else 1)).astype(int)

    groups = _groups(pts, incidents, color_by, slots)
    noise = pts[pts["is_noise"]] if show_noise else pts.iloc[0:0]

    stacks = [(name, color, _stack(sub)) for name, color, sub in groups if len(sub)]
    noise_stack = _stack(noise) if len(noise) else None
    all_counts = [s["n"].max() for _, _, s in stacks]
    if noise_stack is not None:
        all_counts.append(noise_stack["n"].max())
    max_count = max(all_counts) if all_counts else 1

    fig = go.Figure()

    if noise_stack is not None:
        fig.add_trace(go.Scattergl(
            x=noise_stack["x"], y=noise_stack["y"], mode="markers",
            name=f"Unclustered alerts ({len(noise):,})",
            marker=dict(symbol="circle-open", color=NOISE_COLOR, size=_sizes(noise_stack["n"], max_count),
                        line=dict(width=1.5, color=NOISE_COLOR)),
            text=_hover(noise_stack, inc_by_id, noise=True),
            hovertemplate="%{text}<extra></extra>",
        ))

    for name, color, stack in stacks:
        fig.add_trace(go.Scattergl(
            x=stack["x"], y=stack["y"], mode="markers", name=name,
            marker=dict(color=color, size=_sizes(stack["n"], max_count), opacity=0.9,
                        # Surface-coloured ring keeps overlapping markers apart.
                        line=dict(width=1, color=SURFACE)),
            text=_hover(stack, inc_by_id, noise=False),
            hovertemplate="%{text}<extra></extra>",
        ))

    offsets = [(-50, -40), (50, 40), (50, -40)]     # px from a highlighted incident to its boxed label
    highlighted = [i for i in sorted(slots, key=slots.get) if i in inc_by_id]

    # Incident view: name the biggest non-highlighted bubbles too, so the map reads without hovering.
    if by_incident:
        spots = pts[~pts["is_noise"]].groupby("incident_id").agg(x=("x", "first"), y=("y", "first"), n=("x", "size"))
        to_px = np.array([1000, 520]) / np.ptp(pts[["x", "y"]].to_numpy(), axis=0).clip(1e-9)   # approx. plot area
        # Label centres already taken (px, y up), starting with the boxed labels of the highlights.
        taken = [spots.loc[i, ["x", "y"]].to_numpy() * to_px + (offsets[n % 3][0], -offsets[n % 3][1])
                 for n, i in enumerate(highlighted) if i in spots.index]
        for inc_id, r in spots[~spots.index.isin(highlighted) & spots.index.isin(inc_by_id)].nlargest(12, "n").iterrows():
            lift = _sizes(r["n"], max_count) / 2 + 2
            at = np.array([r["x"], r["y"]]) * to_px + (0, lift + 7)
            if any(abs(at[0] - q[0]) < 200 and abs(at[1] - q[1]) < 20 for q in taken):
                continue                      # would print over a label already there
            taken.append(at)
            fig.add_annotation(
                x=r["x"], y=r["y"], showarrow=False, yanchor="bottom", yshift=lift,
                text=html.escape(family_label(inc_by_id[inc_id].get("family"))),
                font=dict(size=10, color=INK, family="IBM Plex Mono"),
            )

    # Direct labels for the highlighted incidents, pinned to each incident's weighted medoid:
    # a real marker in the middle of where most of its alerts are, even if it spans several islands.
    for n, inc_id in enumerate(highlighted):
        sub = pts[pts["incident_id"] == inc_id]
        if sub.empty:
            continue
        cells = sub.groupby(["_bx", "_by"]).agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"))
        cells = cells.nlargest(1500, "n")
        b = cells.index.to_frame().to_numpy(dtype=float)
        dist = np.sqrt(((b[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        anchor = cells.iloc[int((dist * cells["n"].to_numpy()).sum(1).argmin())]
        accent = HIGHLIGHT_COLORS[slots[inc_id]] if color_by == "Incident" else INK
        ax, ay = offsets[n % len(offsets)]
        fig.add_annotation(
            x=anchor["x"], y=anchor["y"], ax=ax, ay=ay,
            text=f"<b>{html.escape(inc_id)}</b> {html.escape(family_label(inc_by_id[inc_id].get('family')))}",
            showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=accent,
            font=dict(size=11, color="#e0e8f0", family="IBM Plex Mono"),
            bgcolor="rgba(15,19,24,0.92)", bordercolor=accent, borderwidth=1, borderpad=4,
        )

    n_clustered = int((~pts["is_noise"]).sum())
    n_noise = int(pts["is_noise"].sum())
    axis_font = dict(size=11, color="#4a8090", family="IBM Plex Mono")
    axis = dict(
        gridcolor="#111820", zeroline=False, showticklabels=show_ticks,
        tickfont=dict(size=9, color="#2a3a4a", family="IBM Plex Mono"), tickformat=".2f",
    )

    # Plotly titles don't wrap, so the key is split into short lines that fit a narrow column.
    key = [
        (f"Bubble = one incident · bigger = more alerts (largest {int(max_count):,})" if by_incident else
         f"Dot = alerts that look the same · bigger = more (largest {int(max_count):,})"),
        ("Closer bubbles = incidents made of more similar alerts" if by_incident else
         "Closer dots = more similar signature, port and protocol"),
        f"Colour = {_COLOR_MEANING.get(color_by, color_by)} · ring = in no incident",
    ]
    fig.update_layout(
        paper_bgcolor="#0a0c10",
        plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Mono", color="#4a6070", size=11),
        margin=dict(l=10, r=10, t=112, b=80),
        xaxis=dict(title=dict(text=x_title, font=axis_font), **axis),
        yaxis=dict(title=dict(text=y_title, font=axis_font), **axis),
        # Horizontal legend under the plot so it never squeezes the map on narrow screens.
        legend=dict(
            orientation="h", x=0, xanchor="left", yref="container", y=0.01, yanchor="bottom",
            bgcolor="#0f1318", bordercolor="#1e2530", borderwidth=1,
            font=dict(size=10, color=INK), itemsizing="constant",
            title=dict(text=f"Colour = {color_by}:", font=dict(size=10, color=MUTED_INK), side="top left"),
        ),
        hoverlabel=dict(bgcolor="#0f1318", bordercolor="#1e2530",
                        font=dict(family="IBM Plex Mono", size=11, color=INK)),
        title=dict(
            text=(
                f"<b>{'Incident' if by_incident else 'Alert'} similarity map</b> · {method}<br>"
                f"{n_clustered:,} alerts in {len(incidents)} incidents · {n_noise:,} unclustered"
                + "".join(f"<br><span style='font-size:11px;color:{MUTED_INK}'>{line}</span>" for line in key)
            ),
            font=dict(size=13, color=INK, family="IBM Plex Mono"),
            x=0.01, xanchor="left", y=0.985, yanchor="top",
        ),
        height=720,
    )
    return fig
