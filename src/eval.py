"""
Clustering benchmark: which algorithm recovers the true incidents best?

Generates a labelled synthetic Suricata log (every alert knows which campaign it
belongs to), runs the real pipeline stages on it once, then scores each clustering
method against the truth with the metrics from the lecture slides:

  purity   - share of alerts sitting in an incident dominated by their own campaign.
             Trivially 1.0 if every alert is its own cluster, so never read it alone.
  inverse purity - share of each campaign that landed in ONE incident (punishes
             fragmenting a campaign); the counterweight to purity.
  ARI      - adjusted Rand index: pairwise "same incident / different incident"
             decisions that are right, corrected for chance. The headline number.
  silhouette - needs no truth; shown to check whether it agrees with the truth
             (it often does not, which is the slides' warning).

Background noise alerts count as one-alert campaigns, and so does every alert a
method leaves unclustered, so swallowing noise and discarding real alerts both cost.

Usage:  python -m src.eval [--seeds 3] [--out eval_out]
"""

import argparse
import contextlib
import io
import json
import os
import random
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

from .cluster import (
    _apply_dbscan, _apply_optics, _postprocess_min_cluster_size, assign_candidate_group,
    cluster_within_groups,
)
from .embed import embed_alerts
from .incidents import build_incidents, stitch_incidents
from .parse_suricata import parse_suricata_alerts

T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
WINDOW_MIN = 30
MIN_SAMPLES = 3

# ── Labelled synthetic data ───────────────────────────────────────────────────
# (name, src_ip, start_min, duration_min, n_alerts, dest_ips, [(signature, category, sev, proto, ports)])
# Deliberately hard cases: several signatures per campaign, scans over many ports,
# campaigns crossing the 30-minute window, and two campaigns from one host at once.
_WEB = [80, 443, 8080]
_NET = lambda p, n: [f"{p}.{i}" for i in range(1, n + 1)]
CAMPAIGNS = [
    ("ssh_brute", "192.168.10.5", 1, 23, 600, _NET("10.1.1", 3), [
        ("ET SCAN Potential SSH Scan", "Attempted Information Leak", 2, "TCP", [22]),
        ("ET BRUTE_FORCE SSH Brute Force Attempt", "Attempted Administrator Privilege Gain", 2, "TCP", [22])]),
    ("rdp_brute", "192.168.10.6", 20, 30, 500, _NET("10.1.2", 2), [
        ("ET BRUTE_FORCE RDP Login Attempt Flood", "Attempted Administrator Privilege Gain", 2, "TCP", [3389])]),
    ("ftp_brute", "192.168.10.7", 62, 20, 250, _NET("10.1.3", 2), [
        ("ET SCAN Potential FTP Brute-Force attempt response", "Unsuccessful User Privilege Gain", 2, "TCP", [21])]),
    ("telnet_brute", "192.168.10.8", 95, 15, 200, _NET("10.1.4", 4), [
        ("ET SCAN Suspicious inbound to Telnet port 23", "Potentially Bad Traffic", 2, "TCP", [23]),
        ("ET TELNET busybox MIRAI hackers default credentials", "Attempted Administrator Privilege Gain", 1, "TCP", [23])]),
    ("nmap_scan", "192.168.20.1", 2, 20, 1000, _NET("10.2.0", 50), [
        ("ET SCAN NMAP -sS window 1024", "Attempted Information Leak", 2, "TCP", list(range(1, 1025))),
        ("ET SCAN Possible Nmap User-Agent Observed", "Web Application Attack", 1, "TCP", _WEB)]),
    ("ping_sweep", "192.168.20.2", 31, 10, 150, _NET("10.2.1", 100), [
        ("ET SCAN ICMP Ping Sweep", "Misc activity", 3, "ICMP", [None])]),
    # One attacker, recon then exploitation in the same window: two incidents.
    ("recon_before_log4j", "185.220.101.1", 31, 8, 200, _NET("10.3.0", 20), [
        ("ET SCAN NMAP -sS window 1024", "Attempted Information Leak", 2, "TCP", list(range(1, 1025)))]),
    ("log4j", "185.220.101.1", 40, 15, 400, _NET("10.3.0", 3), [
        ("ET EXPLOIT Apache log4j RCE Attempt (http ldap) (CVE-2021-44228)", "Attempted Administrator Privilege Gain", 1, "TCP", _WEB),
        ("ET EXPLOIT Apache log4j RCE Attempt - lower/upper TCP Bypass (CVE-2021-44228)", "Attempted Administrator Privilege Gain", 1, "TCP", _WEB)]),
    ("sqli", "185.220.101.5", 61, 25, 600, _NET("10.3.1", 2), [
        ("ET WEB_SERVER Possible SQL Injection Attempt UNION SELECT", "Web Application Attack", 1, "TCP", _WEB),
        ("ET WEB_SERVER Possible SQL Injection Attempt SELECT FROM", "Web Application Attack", 1, "TCP", _WEB),
        ("ET SCAN Sqlmap SQL Injection Scan", "Attempted Information Leak", 2, "TCP", _WEB)]),
    ("xss", "185.220.101.6", 91, 20, 300, _NET("10.3.2", 2), [
        ("ET WEB_SERVER Script tag in URI Possible Cross Site Scripting Attempt", "Web Application Attack", 1, "TCP", [80, 443])]),
    ("shellshock", "45.33.32.156", 122, 20, 300, _NET("10.3.3", 5), [
        ("ET WEB_SERVER Possible CVE-2014-6271 Attempt in Headers", "Attempted Administrator Privilege Gain", 1, "TCP", [80]),
        ("ET WEB_SERVER Possible CVE-2014-6271 Attempt", "Attempted Administrator Privilege Gain", 1, "TCP", [80])]),
    # Long-running C2 crossing two window boundaries.
    ("cobalt_strike", "10.0.0.77", 60, 70, 800, ["91.108.4.1"], [
        ("ET MALWARE Cobalt Strike Beacon Detected", "A Network Trojan was detected", 1, "TCP", [443]),
        ("ET MALWARE Cobalt Strike Malleable C2 Profile (jquery)", "A Network Trojan was detected", 1, "TCP", [443]),
        ("ET JA3 Hash - [Abuse.ch] Possible Cobalt Strike", "Unknown Traffic", 3, "TCP", [443])]),
    # ...and the same infected host mining at the same time: a second incident.
    ("cryptomining", "10.0.0.77", 65, 20, 250, ["51.15.58.224"], [
        ("ET POLICY Cryptocurrency Miner Checkin", "Potential Corporate Privacy Violation", 2, "TCP", [3333]),
        ("ET COINMINER XMR CoinMiner Usage", "Crypto Currency Mining Activity Detected", 2, "TCP", [3333, 14444])]),
    ("emotet", "10.0.0.78", 121, 25, 500, ["45.77.1.9", "103.4.5.6"], [
        ("ET MALWARE Win32/Emotet CnC Activity (POST) M10", "A Network Trojan was detected", 1, "TCP", [443, 8080, 7080]),
        ("ET CNC Feodo Tracker Reported CnC Server group 3", "A Network Trojan was detected", 1, "TCP", [443, 8080, 7080])]),
    ("slow_beacon", "10.0.0.60", 0, 180, 40, ["198.51.100.7"], [
        ("ET MALWARE Cobalt Strike Beacon Detected", "A Network Trojan was detected", 1, "TCP", [8443])]),
    ("eternalblue", "10.0.0.20", 3, 22, 600, _NET("10.0.1", 40), [
        ("ET EXPLOIT ETERNALBLUE Exploit M2 MS17-010", "Attempted Administrator Privilege Gain", 1, "TCP", [445]),
        ("ET EXPLOIT Possible ETERNALBLUE Probe MS17-010 (MSF style)", "Attempted Administrator Privilege Gain", 1, "TCP", [445]),
        ("ET SCAN Behavioral Unusual Port 445 traffic Potential Scan or Infection", "Misc activity", 3, "TCP", [445])]),
    ("smb_lateral", "10.0.0.21", 33, 20, 250, _NET("10.0.2", 6), [
        ("ET POLICY Powershell Activity Over SMB - Likely Lateral Movement", "Potential Corporate Privacy Violation", 1, "TCP", [445]),
        ("ET POLICY Powershell Command With Hidden Window Argument Over SMB - Likely Lateral Movement", "Potential Corporate Privacy Violation", 1, "TCP", [445]),
        ("ET POLICY SMB2 NT Create AndX Request For an Executable File", "Potentially Bad Traffic", 2, "TCP", [445]),
        ("ET POLICY SMB Remote AT Scheduled Job Pipe Creation", "Potentially Bad Traffic", 2, "TCP", [445])]),
    ("printnightmare", "10.0.0.22", 63, 20, 300, _NET("10.0.3", 3), [
        ("ET EXPLOIT Possible PrintNightmare RPC Call (CVE-2021-1675)", "Attempted Administrator Privilege Gain", 1, "TCP", [445, 135])]),
    ("dns_dga", "192.168.30.1", 92, 25, 400, ["8.8.8.8", "1.1.1.1"], [
        ("ET DNS Query to a *.top domain - Likely Hostile", "Potentially Bad Traffic", 2, "UDP", [53]),
        ("ET DNS Query for Suspicious .pw Domain DGA", "Potentially Bad Traffic", 2, "UDP", [53])]),
    ("exfil", "10.0.0.88", 123, 20, 300, ["203.0.113.50"], [
        ("ET POLICY Possible Large Outbound FTP Data Transfer", "Potential Corporate Privacy Violation", 2, "TCP", [21, 20])]),
    # Real ET Tor rules carry a group number, so the signature text itself varies.
    ("tor", "10.0.0.99", 151, 25, 200, _NET("171.25.193", 30), [
        (f"ET TOR Known Tor Relay/Router (Not Exit) Node Traffic group {g}", "Misc Attack", 2, "TCP", [9001, 443])
        for g in range(1, 40)]),
    ("apt_updates", "10.0.0.30", 5, 10, 300, ["91.189.91.38"], [
        ("ET POLICY GNU/Linux APT User-Agent Outbound likely related to package management", "Not Suspicious Traffic", 3, "TCP", [80])]),
]

NOISE_SIGS = [
    ("SURICATA STREAM 3way handshake wrong seq wrong ack", "Generic Protocol Command Decode", 3, "TCP", [443, 80, 8443]),
    ("SURICATA TLS invalid record/traffic", "Generic Protocol Command Decode", 3, "TCP", [443]),
    ("SURICATA HTTP unable to match response to request", "Generic Protocol Command Decode", 3, "TCP", [80]),
    ("ET POLICY Dropbox.com Offsite File Backup in Use", "Potential Corporate Privacy Violation", 1, "TCP", [443]),
    ("ET INFO Windows OS Submitting USB Metadata to Microsoft", "Misc activity", 3, "TCP", [80]),
    ("ET POLICY PE EXE or DLL Windows file download HTTP", "Potential Corporate Privacy Violation", 1, "TCP", [80]),
    ("ET INFO TLS Handshake Failure", "Potentially Bad Traffic", 2, "TCP", [443, 993]),
    ("ET POLICY Outgoing Basic Auth Base64 HTTP Password detected unencrypted", "Potential Corporate Privacy Violation", 1, "TCP", [80, 8080]),
    ("ET INFO Observed DNS Query to .cloud TLD", "Potentially Bad Traffic", 2, "UDP", [53]),
    ("ET POLICY Python-urllib/ Suspicious User Agent", "Attempted Information Leak", 2, "TCP", [80, 8000]),
    ("ET CHAT IRC NICK command", "Misc activity", 3, "TCP", [6667]),
    ("ET INFO Session Traversal Utilities for NAT (STUN Binding Request)", "Misc activity", 3, "UDP", [3478]),
    ("SURICATA SMTP tls rejected", "Generic Protocol Command Decode", 3, "TCP", [25, 587]),
    ("ET POLICY SSLv3 inbound connection to server vulnerable to POODLE attack", "Potential Corporate Privacy Violation", 1, "TCP", [443]),
    ("ET INFO NTP Client Request", "Misc activity", 3, "UDP", [123]),
    ("GPL SNMP public access udp", "Attempted Information Leak", 2, "UDP", [161]),
]


def write_labelled_eve(path: str, seed: int = 0) -> list:
    """Write a synthetic eve.json; return the true campaign of each line (None = background noise)."""
    rng = random.Random(seed)
    events = []

    def emit(minute, src, dst, sig, truth):
        name, cat, sev, proto, ports = sig
        port = rng.choice(ports)
        ev = {
            "timestamp": (T0 + timedelta(minutes=minute)).strftime("%Y-%m-%dT%H:%M:%S.%f+0000"),
            "event_type": "alert", "src_ip": src, "dest_ip": dst, "proto": proto,
            "flow_id": rng.getrandbits(48),
            "alert": {"signature": name, "category": cat, "severity": sev},
        }
        if port is not None:
            ev["src_port"], ev["dest_port"] = rng.randint(32768, 60999), port
        events.append((ev["timestamp"], ev, truth))

    for name, src, start, dur, n, dests, sigs in CAMPAIGNS:
        weights = [1 / (i + 1) for i in range(len(sigs))]   # first signature dominates
        for _ in range(n):
            emit(start + rng.random() * dur, src, rng.choice(dests), rng.choices(sigs, weights)[0], name)
        # Infected / attacking hosts still produce unrelated background alerts.
        for _ in range(12):
            emit(rng.random() * 180, src, f"172.16.{rng.randint(0, 9)}.{rng.randint(1, 250)}", rng.choice(NOISE_SIGS), None)

    for _ in range(600):   # one-off alerts from hosts that are not attacking anything
        emit(rng.random() * 180, f"10.9.{rng.randint(0, 3)}.{rng.randint(1, 120)}",
             f"172.16.{rng.randint(0, 9)}.{rng.randint(1, 250)}", rng.choice(NOISE_SIGS), None)

    events.sort(key=lambda e: e[0])
    with open(path, "w", encoding="utf-8") as f:
        for _, ev, _ in events:
            f.write(json.dumps(ev) + "\n")
    return [t for _, _, t in events]


# ── Clustering methods: embeddings of one candidate group -> labels (-1 = noise) ─

def _hier(linkage):
    def fit(X, t):
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=t, linkage=linkage, metric="cosine").fit_predict(X)
        return _postprocess_min_cluster_size(labels, MIN_SAMPLES)
    return fit


def _n_distinct(X):
    return len(np.unique(X.round(5), axis=0))


def _kmeans(X, min_sil):
    """K-means, K picked by silhouette. K=1 has no silhouette, so it needs a floor to fall back on."""
    best, best_labels = min_sil, np.zeros(len(X), dtype=int)
    for k in range(2, min(10, _n_distinct(X) - 1) + 1):
        labels = KMeans(k, n_init=5, random_state=0).fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine", sample_size=min(len(X), 1500), random_state=0)
        if score > best:
            best, best_labels = score, labels
    return _postprocess_min_cluster_size(best_labels, MIN_SAMPLES)


def _gmm(X, reg):
    """Gaussian mixture on a PCA reduction (384-d covariances are not estimable), K picked by BIC."""
    d = _n_distinct(X)
    if d < 2:
        return np.zeros(len(X), dtype=int)
    Z = PCA(min(10, d - 1, len(X) - 1), random_state=0).fit_transform(X)
    fits = [GaussianMixture(k, covariance_type="diag", reg_covar=reg, random_state=0).fit(Z)
            for k in range(1, min(10, d) + 1)]
    labels = min(fits, key=lambda m: m.bic(Z)).predict(Z)
    return _postprocess_min_cluster_size(labels, MIN_SAMPLES)


def _spectral(X, sigma):
    """Spectral clustering on a Gaussian similarity graph, K picked by the eigengap."""
    D = np.clip(1 - X @ X.T, 0, None)
    W = np.exp(-(D ** 2) / (2 * sigma ** 2))
    deg = W.sum(1)
    L = np.eye(len(X)) - W / np.sqrt(np.outer(deg, deg))
    vals = np.linalg.eigvalsh(L)[:11]
    k = int(np.argmax(np.diff(vals))) + 1
    if k == 1:
        return np.zeros(len(X), dtype=int)
    labels = SpectralClustering(k, affinity="precomputed", random_state=0).fit_predict(W)
    return _postprocess_min_cluster_size(labels, MIN_SAMPLES)


def _hdbscan(X, min_cluster_size):
    if len(X) < min_cluster_size:
        return np.full(len(X), -1)
    return HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine",
                   allow_single_cluster=True, copy=True).fit_predict(X)


# (label, function, parameter name, values to sweep)
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
METHODS = [
    ("DBSCAN", lambda X, e: _apply_dbscan(X, e, MIN_SAMPLES), "eps", THRESHOLDS),
    ("Agglomerative (average link)", _hier("average"), "threshold", THRESHOLDS),
    ("Agglomerative (complete link)", _hier("complete"), "threshold", THRESHOLDS),
    ("Agglomerative (single link)", _hier("single"), "threshold", THRESHOLDS),
    ("OPTICS", lambda X, xi: _apply_optics(X, MIN_SAMPLES, xi, MIN_SAMPLES), "xi", [0.01, 0.05, 0.1, 0.3]),
    ("HDBSCAN", _hdbscan, "min_cluster_size", [3, 10, 30, 100]),
    ("K-means (K by silhouette)", _kmeans, "min silhouette", [0.3, 0.5, 0.7, 0.9]),
    ("Gaussian mixture (K by BIC)", _gmm, "reg_covar", [1e-6, 1e-4, 1e-3, 1e-2]),
    ("Spectral (K by eigengap)", _spectral, "sigma", [0.05, 0.1, 0.2, 0.3]),
]


def cluster_groups(df: pd.DataFrame, E: np.ndarray, fit) -> np.ndarray:
    """Same per-candidate-group loop as cluster.cluster_within_groups, for any method."""
    out = np.full(len(df), -1)
    offset = 0
    for idx in df.groupby("candidate_group").indices.values():
        if len(idx) < MIN_SAMPLES:
            continue
        labels = np.asarray(fit(E[idx]))
        out[idx] = np.where(labels == -1, -1, labels + offset)
        offset += int(labels.max()) + 1 if (labels >= 0).any() else 0
    return out


def to_incidents(df: pd.DataFrame, cluster_ids: np.ndarray) -> np.ndarray:
    """Run the app's incident building + stitching; return an incident number per alert (-1 = none)."""
    d = df.assign(cluster_id=cluster_ids)
    with contextlib.redirect_stdout(io.StringIO()):     # stitch_incidents prints debug lines
        incidents = stitch_incidents(build_incidents(d))
    lookup = {c: n for n, inc in enumerate(incidents) for c in inc.get("merged_cluster_ids", [inc["cluster_id"]])}
    return np.array([lookup.get(int(c), -1) for c in cluster_ids])


# ── Scoring ───────────────────────────────────────────────────────────────────

def _singletons(labels: np.ndarray) -> np.ndarray:
    """Give every noise alert (-1) its own id so it counts as a cluster of one."""
    out = labels.copy()
    noise = out == -1
    out[noise] = out.max() + 1 + np.arange(noise.sum())
    return out


def _purity(a: np.ndarray, b: np.ndarray) -> float:
    """Share of items whose cluster in `a` is dominated by their class in `b`."""
    return pd.crosstab(a, b).max(axis=1).sum() / len(a)


def score(truth: np.ndarray, pred: np.ndarray, df: pd.DataFrame, E: np.ndarray) -> dict:
    t, p = _singletons(truth), _singletons(pred)
    # Silhouette only exists inside a candidate group that was split into >= 2 clusters.
    sil = []
    for idx in df.groupby("candidate_group").indices.values():
        lab = pred[df.index[idx]]
        keep = lab != -1
        if len(set(lab[keep])) >= 2 and keep.sum() > len(set(lab[keep])):
            sil.extend(silhouette_samples(E[idx][keep], lab[keep], metric="cosine"))
    return {
        "ARI": adjusted_rand_score(t, p),
        "Rand": rand_score(t, p),
        "purity": _purity(p, t),
        "inv_purity": _purity(t, p),
        "incidents": len(set(pred) - {-1}),
        "true_incidents": len(set(truth) - {-1}),
        "noise_kept_out": float((pred[truth == -1] == -1).mean()),
        "alerts_dropped": float((pred[truth != -1] == -1).mean()),
        "silhouette": float(np.mean(sil)) if sil else float("nan"),
    }


def run_seed(seed: int, workdir: str) -> pd.DataFrame:
    eve = os.path.join(workdir, f"labelled_eve_seed{seed}.json")
    truth_names = write_labelled_eve(eve, seed)
    parsed = parse_suricata_alerts(eve)
    # alert_id ends with the alert's line number, which indexes the truth list.
    codes = {n: i for i, n in enumerate(c[0] for c in CAMPAIGNS)}
    truth = np.array([codes.get(truth_names[int(a.rsplit("-", 1)[1])], -1) for a in parsed["alert_id"]])

    # Same pre-filter as pipeline.run_pipeline; alerts it drops stay "unclustered".
    grouped = assign_candidate_group(parsed, WINDOW_MIN)
    big = grouped["candidate_group"].map(grouped["candidate_group"].value_counts()) >= MIN_SAMPLES
    df = grouped[big]
    E = embed_alerts(df)
    df = df.reset_index().rename(columns={"index": "row"})   # row = position in `parsed`
    E_all = np.zeros((len(parsed), E.shape[1]), dtype="float32")
    E_all[df["row"]] = E
    full = grouped.assign(cluster_id=-1)

    # Self-check of the app's own code path: an incident must never span two candidate groups.
    with contextlib.redirect_stderr(io.StringIO()):     # tqdm bar
        app = cluster_within_groups(df, E, algo="hierarchical", distance_threshold=0.35, min_samples=MIN_SAMPLES)
    app = app[app["cluster_id"] != -1]
    assert app.groupby("cluster_id")["candidate_group"].nunique().max() == 1, "cluster id reused across groups"

    rows = []
    for name, fit, pname, values in METHODS:
        for v in values:
            pred = np.full(len(parsed), -1)
            pred[df["row"]] = to_incidents(df, cluster_groups(df, E, lambda X: fit(X, v)))
            rows.append({"method": name, "param": pname, "value": v, "seed": seed,
                         **score(truth, pred, full, E_all)})
            print(f"  seed {seed}  {name:32s} {pname}={v:<6}  ARI={rows[-1]['ARI']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Benchmark clustering methods on labelled synthetic alerts")
    ap.add_argument("--seeds", type=int, default=3, help="number of synthetic datasets to average over")
    ap.add_argument("--only", default="", help="run only methods whose name contains this text, e.g. Agglomerative")
    ap.add_argument("--out", default="eval_out", help="directory for the generated logs and results.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    warnings.filterwarnings("ignore")   # OPTICS / spectral complain about duplicate points

    METHODS[:] = [m for m in METHODS if args.only.lower() in m[0].lower()]
    res = pd.concat([run_seed(s, args.out) for s in range(args.seeds)])
    metrics = [c for c in res.columns if c not in ("method", "param", "value", "seed")]
    mean = res.groupby(["method", "param", "value"], sort=False)[metrics].mean().reset_index()
    mean.to_csv(os.path.join(args.out, "results.csv"), index=False)

    best = mean.loc[mean.groupby("method", sort=False)["ARI"].idxmax()].sort_values("ARI", ascending=False)
    pd.set_option("display.width", 250)
    print(f"\nBest setting per method, mean of {args.seeds} dataset(s), ranked by ARI:\n")
    print(best.round(3).to_string(index=False))
    print(f"\nFull sweep: {os.path.join(args.out, 'results.csv')}")


if __name__ == "__main__":
    main()
