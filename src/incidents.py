from typing import Any, Dict, List

import pandas as pd


def template_summary(g: pd.DataFrame) -> str:
    src = g["src_ip"].iloc[0]
    dests = g["dest_ip"].value_counts().head(3).index.tolist()
    sig = g["signature"].value_counts().idxmax()
    count = len(g)
    ports = sorted(set([int(p) for p in g["dest_port"].dropna().tolist()]))[:10]

    dest_str = ", ".join(dests)
    port_str = ", ".join(map(str, ports)) if ports else "unknown"
    return f"{count} related alerts likely tied to '{sig}' from {src} targeting {dest_str} on ports {port_str}."


def build_incidents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []

    clustered = df[df["cluster_id"] != -1].copy()
    if clustered.empty:
        return incidents

    for cid, g in clustered.groupby("cluster_id"):
        g = g.sort_values("ts_dt")
        start = g["ts_dt"].iloc[0]
        end = g["ts_dt"].iloc[-1]

        top_sigs = g["signature"].value_counts().head(3).to_dict()
        max_sev = int(g["severity"].max()) if g["severity"].notna().any() else None

        incident = {
            "incident_id": f"INC-{int(cid):05d}",
            "cluster_id": int(cid),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "src_ips": sorted(set(g["src_ip"].dropna().astype(str).tolist())),
            "dest_ips": sorted(set(g["dest_ip"].dropna().astype(str).tolist())),
            "dest_ports": sorted(set([int(p) for p in g["dest_port"].dropna().tolist()])),
            "proto": sorted(set(g["proto"].dropna().astype(str).tolist())),
            "alert_count": int(len(g)),
            "top_signatures": top_sigs,
            "max_severity": max_sev,
            "representative_alerts": g[["timestamp", "src_ip", "dest_ip", "signature", "category", "severity"]]
            .head(5)
            .to_dict(orient="records"),
            "summary": template_summary(g),
        }
        incidents.append(incident)

    incidents.sort(key=lambda x: (x["start_time"], -(x["max_severity"] or 0), -x["alert_count"]))
    return incidents
