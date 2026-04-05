
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd


FAMILY_RULES: List[Tuple[str, str]] = [
    ("cobalt strike", "malware_c2"),
    ("malleable c2", "malware_c2"),
    ("beacon detected", "malware_c2"),
    ("ms17-010", "smb_worm"),
    ("eternalblue", "smb_worm"),
    ("log4j", "java_rce"),
    ("sql injection", "web_sqli"),
    ("request smuggling", "web_exploitation"),
    ("script tag", "web_exploitation"),
    ("xss", "web_exploitation"),
    ("ssh brute force", "ssh_bruteforce"),
    ("rdp login attempt", "rdp_bruteforce"),
    ("nmap", "recon_scan"),
    ("ping sweep", "recon_scan"),
    ("dga", "dns_suspicious"),
    ("suspicious tld", "dns_suspicious"),
]


def normalize_family(signatures: List[str], categories: List[str], dest_ports: List[int]) -> str:
    blob = " | ".join(signatures + categories).lower()
    for needle, family in FAMILY_RULES:
        if needle in blob:
            return family

    port_set = set(dest_ports)
    if 22 in port_set:
        return "ssh_activity"
    if 3389 in port_set:
        return "rdp_activity"
    if 445 in port_set:
        return "smb_activity"
    if 53 in port_set:
        return "dns_activity"
    if port_set & {80, 443, 8080, 8443}:
        return "web_activity"
    return "misc_activity"


def build_summary_from_family(
    family: str,
    src_ips: List[str],
    dest_ips: List[str],
    dest_ports: List[int],
    alert_count: int,
    top_signature: str,
) -> str:
    src = ", ".join(src_ips[:2])
    dst = ", ".join(dest_ips[:3]) + ("…" if len(dest_ips) > 3 else "")
    ports = ", ".join(map(str, dest_ports[:8])) if dest_ports else "unknown"

    templates = {
        "ssh_bruteforce": f"{alert_count} alerts indicate an SSH brute-force campaign from {src} against {dst} on port 22.",
        "rdp_bruteforce": f"{alert_count} alerts indicate an RDP brute-force campaign from {src} against {dst} on port 3389.",
        "malware_c2": f"{alert_count} alerts indicate repeated malware command-and-control or beaconing activity from {src} to {dst} on ports {ports}.",
        "smb_worm": f"{alert_count} alerts indicate SMB exploitation or lateral movement from {src} against {dst} on port 445.",
        "java_rce": f"{alert_count} alerts indicate repeated Java RCE exploitation attempts from {src} against {dst} on ports {ports}.",
        "web_sqli": f"{alert_count} alerts indicate a SQL injection campaign from {src} against {dst} on ports {ports}.",
        "web_exploitation": f"{alert_count} alerts indicate a web exploitation campaign from {src} against {dst} on ports {ports}.",
        "recon_scan": f"{alert_count} alerts indicate reconnaissance or network scanning from {src} against {dst} on ports {ports}.",
        "dns_suspicious": f"{alert_count} alerts indicate suspicious DNS activity from {src} to {dst} on port 53.",
    }
    return templates.get(
        family,
        f"{alert_count} related alerts likely tied to '{top_signature}' from {src} targeting {dst} on ports {ports}.",
    )


def compute_risk_score(max_severity: int | None, family: str, alert_count: int) -> int:
    sev_score = {1: 40, 2: 25, 3: 12, 4: 5, None: 0}[max_severity]
    family_bonus = {
        "malware_c2": 30,
        "smb_worm": 28,
        "java_rce": 26,
        "web_sqli": 22,
        "web_exploitation": 18,
        "ssh_bruteforce": 14,
        "rdp_bruteforce": 16,
        "recon_scan": 8,
        "dns_suspicious": 10,
    }.get(family, 0)
    volume_bonus = min(20, int(alert_count / 5))
    return sev_score + family_bonus + volume_bonus


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
        max_sev = int(g["severity"].min()) if g["severity"].notna().any() else None  # lower is higher severity
        src_ips = sorted(set(g["src_ip"].dropna().astype(str).tolist()))
        dest_ips = sorted(set(g["dest_ip"].dropna().astype(str).tolist()))
        dest_ports = sorted(set([int(p) for p in g["dest_port"].dropna().tolist()]))
        proto = sorted(set(g["proto"].dropna().astype(str).tolist()))
        categories = sorted(set(g["category"].dropna().astype(str).tolist()))
        family = normalize_family(list(top_sigs.keys()), categories, dest_ports)
        top_sig = next(iter(top_sigs.keys()))

        incident = {
            "incident_id": f"INC-{int(cid):05d}",
            "cluster_id": int(cid),
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "src_ips": src_ips,
            "dest_ips": dest_ips,
            "dest_ports": dest_ports,
            "proto": proto,
            "alert_count": int(len(g)),
            "top_signatures": top_sigs,
            "max_severity": max_sev,
            "family": family,
            "risk_score": compute_risk_score(max_sev, family, int(len(g))),
            "representative_alerts": g[["timestamp", "src_ip", "dest_ip", "signature", "category", "severity"]]
            .head(5)
            .to_dict(orient="records"),
            "summary": build_summary_from_family(family, src_ips, dest_ips, dest_ports, int(len(g)), top_sig),
        }
        incidents.append(incident)

    incidents.sort(key=lambda x: (x["start_time"], -x["risk_score"], -x["alert_count"]))
    return incidents


def _top_signature_overlap(a: Dict[str, int], b: Dict[str, int]) -> float:
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    if not a_keys or not b_keys:
        return 0.0
    return len(a_keys & b_keys) / len(a_keys | b_keys)


def _should_merge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    a_end = datetime.fromisoformat(a["end_time"])
    b_start = datetime.fromisoformat(b["start_time"])
    gap = max(timedelta(seconds=0), b_start - a_end)

    a_src = set(map(str, a.get("src_ips", [])))
    b_src = set(map(str, b.get("src_ips", [])))
    a_dst = set(map(str, a.get("dest_ips", [])))
    b_dst = set(map(str, b.get("dest_ips", [])))
    a_ports = set(map(int, a.get("dest_ports", [])))
    b_ports = set(map(int, b.get("dest_ports", [])))

    a_family = str(a.get("family", "")).strip().lower()
    b_family = str(b.get("family", "")).strip().lower()

    same_src = len(a_src & b_src) > 0
    same_family = a_family == b_family
    same_dest = len(a_dst & b_dst) > 0
    same_port = len(a_ports & b_ports) > 0

    # Debug
    print(
        "DEBUG SHOULD_MERGE:",
        f"a={a.get('cluster_id')} b={b.get('cluster_id')}",
        f"family={a_family}/{b_family}",
        f"same_src={same_src}",
        f"same_dest={same_dest}",
        f"same_port={same_port}",
        f"gap={gap}"
    )

    # Long-running campaigns
    if same_family and same_src and gap <= timedelta(minutes=30):
        if a_family in {"malware_c2", "smb_worm", "ssh_bruteforce", "rdp_bruteforce"}:
            if same_dest or same_port:
                return True

    # Recon fragments
    if same_family and same_src and a_family == "recon_scan" and gap <= timedelta(minutes=15):
        return True

    return False


def _merge_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    start_time = min(a["start_time"], b["start_time"])
    end_time = max(a["end_time"], b["end_time"])
    merged_sigs: Dict[str, int] = {}
    for src in (a["top_signatures"], b["top_signatures"]):
        for sig, count in src.items():
            merged_sigs[sig] = merged_sigs.get(sig, 0) + count

    src_ips = sorted(set(a["src_ips"]) | set(b["src_ips"]))
    dest_ips = sorted(set(a["dest_ips"]) | set(b["dest_ips"]))
    dest_ports = sorted(set(a["dest_ports"]) | set(b["dest_ports"]))
    proto = sorted(set(a["proto"]) | set(b["proto"]))
    max_sev = min(v for v in [a.get("max_severity"), b.get("max_severity")] if v is not None) if (a.get("max_severity") is not None or b.get("max_severity") is not None) else None
    alert_count = a["alert_count"] + b["alert_count"]
    family = a.get("family") or b.get("family")
    top_sig = max(merged_sigs.items(), key=lambda kv: kv[1])[0]

    merged = {
        "incident_id": a["incident_id"],
        "cluster_id": a["cluster_id"],
        "start_time": start_time,
        "end_time": end_time,
        "src_ips": src_ips,
        "dest_ips": dest_ips,
        "dest_ports": dest_ports,
        "proto": proto,
        "alert_count": alert_count,
        "top_signatures": dict(sorted(merged_sigs.items(), key=lambda kv: kv[1], reverse=True)[:3]),
        "max_severity": max_sev,
        "family": family,
        "risk_score": compute_risk_score(max_sev, family, alert_count),
        "representative_alerts": (a.get("representative_alerts", []) + b.get("representative_alerts", []))[:5],
        "summary": build_summary_from_family(family, src_ips, dest_ips, dest_ports, alert_count, top_sig),
        "merged_cluster_ids": sorted(set(a.get("merged_cluster_ids", [a["cluster_id"]])) | set(b.get("merged_cluster_ids", [b["cluster_id"]]))),
    }
    return merged


def stitch_incidents(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    print("DEBUG STITCH FILE:", __file__)
    print("DEBUG STITCH INCIDENT COUNT IN:", len(incidents))

    if not incidents:
        return incidents

    incidents = sorted(incidents, key=lambda x: x["start_time"])
    stitched: List[Dict[str, Any]] = []

    for inc in incidents:
        merged = False

        for j in range(len(stitched) - 1, -1, -1):
            print("DEBUG CHECK:", stitched[j].get("cluster_id"), "vs", inc.get("cluster_id"))
            if _should_merge(stitched[j], inc):
                print("DEBUG MERGING:", stitched[j].get("cluster_id"), "with", inc.get("cluster_id"))
                stitched[j] = _merge_pair(stitched[j], inc)
                merged = True
                break

        if not merged:
            inc = dict(inc)
            inc["merged_cluster_ids"] = [inc["cluster_id"]]
            stitched.append(inc)

    stitched.sort(key=lambda x: x["start_time"])
    for i, inc in enumerate(stitched):
        inc["incident_id"] = f"INC-{i:05d}"

    return stitched