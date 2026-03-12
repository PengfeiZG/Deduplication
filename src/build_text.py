from typing import Optional


def is_private_ip(ip: Optional[str]) -> bool:
    if not ip:
        return False
    return (
        ip.startswith("10.")
        or ip.startswith("192.168.")
        or ip.startswith("172.16.")
        or ip.startswith("172.17.")
        or ip.startswith("172.18.")
        or ip.startswith("172.19.")
        or ip.startswith("172.2")
        or ip.startswith("172.30.")
        or ip.startswith("172.31.")
    )


def port_bucket(port: Optional[int]) -> str:
    if port is None:
        return "unknown"
    if port in (22, 23, 3389):
        return "admin"
    if port in (80, 443, 8080, 8443):
        return "web"
    if port in (53,):
        return "dns"
    if port in (25, 110, 143, 465, 587, 993, 995):
        return "mail"
    if port in (139, 445):
        return "smb"
    return "other"


def build_alert_text(
    signature: str,
    category: Optional[str],
    severity: Optional[int],
    proto: Optional[str],
    src_port: Optional[int],
    dest_port: Optional[int],
    src_ip: Optional[str],
    dest_ip: Optional[str],
) -> str:
    tokens = []
    tokens.append(f"SIG={signature}")
    if category:
        tokens.append(f"CAT={category}")
    if severity is not None:
        tokens.append(f"SEV={severity}")
    if proto:
        tokens.append(f"PROTO={proto}")
    if src_port is not None:
        tokens.append(f"SRC_PORT={src_port}")
        tokens.append(f"SRC_BUCKET={port_bucket(src_port)}")
    if dest_port is not None:
        tokens.append(f"DST_PORT={dest_port}")
        tokens.append(f"DST_BUCKET={port_bucket(dest_port)}")
    if src_ip:
        tokens.append(f"SRC_PRIV={is_private_ip(src_ip)}")
    if dest_ip:
        tokens.append(f"DST_PRIV={is_private_ip(dest_ip)}")

    return " ".join(tokens)
