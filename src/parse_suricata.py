import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .build_text import build_alert_text


@dataclass
class AlertRecord:
    alert_id: str
    timestamp: str
    src_ip: str
    src_port: Optional[int]
    dest_ip: str
    dest_port: Optional[int]
    proto: Optional[str]
    flow_id: Optional[str]
    signature: str
    category: Optional[str]
    severity: Optional[int]
    alert_text: str


def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return None


def parse_suricata_alerts(eve_json_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    count = 0

    for ev in iter_json_lines(eve_json_path):
        if ev.get("event_type") != "alert":
            continue

        ts = ev.get("timestamp")
        if not ts:
            continue

        alert = ev.get("alert", {}) or {}
        signature = alert.get("signature") or "UNKNOWN_SIGNATURE"
        category = alert.get("category")
        severity = _to_int_or_none(alert.get("severity"))

        src_ip = ev.get("src_ip") or ""
        dest_ip = ev.get("dest_ip") or ""
        src_port = _to_int_or_none(ev.get("src_port"))
        dest_port = _to_int_or_none(ev.get("dest_port"))
        proto = str(ev.get("proto")) if ev.get("proto") is not None else None
        flow_id = str(ev.get("flow_id")) if ev.get("flow_id") is not None else None

        text = build_alert_text(
            signature=str(signature),
            category=str(category) if category is not None else None,
            severity=severity,
            proto=proto,
            src_port=src_port,
            dest_port=dest_port,
            src_ip=src_ip,
            dest_ip=dest_ip,
        )

        alert_id = f"{ts}-{flow_id or ''}-{count}"

        rec = AlertRecord(
            alert_id=alert_id,
            timestamp=ts,
            src_ip=src_ip,
            src_port=src_port,
            dest_ip=dest_ip,
            dest_port=dest_port,
            proto=proto,
            flow_id=flow_id,
            signature=str(signature),
            category=str(category) if category is not None else None,
            severity=severity,
            alert_text=text,
        )
        rows.append(asdict(rec))

        count += 1
        if limit and count >= limit:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Suricata alert events found. Confirm eve.json contains event_type=alert lines.")
    return df
