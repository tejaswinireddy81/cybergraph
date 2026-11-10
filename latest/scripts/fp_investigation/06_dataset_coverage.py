"""[fp_investigation] How well does the training data's BENIGN class cover modern traffic types?

Read-only. Prints attack-family counts, benign coverage by traffic type, capture span.
"""
import os
import numpy as np  # noqa: F401
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
df = pd.read_csv(os.path.join(ROOT, "NF-UNSW-NB15-v3.csv"),
                 usecols=["Attack", "PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT", "FLOW_START_MILLISECONDS"])
a = df.Attack.astype(str).str.strip()
print("Attack families (flows):")
print(a.value_counts().to_string())
b = df[a.str.lower() == "benign"]
print("\nBENIGN flows:", len(b))


def port(d, p):
    return (d.L4_DST_PORT == p) | (d.L4_SRC_PORT == p)


rows = [
    ("TCP/443 (HTTPS/TLS)", (b.PROTOCOL == 6) & port(b, 443)),
    ("TCP/80 (HTTP)", (b.PROTOCOL == 6) & port(b, 80)),
    ("TCP other", (b.PROTOCOL == 6) & ~port(b, 443) & ~port(b, 80)),
    ("UDP/53 (DNS)", (b.PROTOCOL == 17) & port(b, 53)),
    ("TCP/53", (b.PROTOCOL == 6) & port(b, 53)),
    ("UDP/443 (QUIC)", (b.PROTOCOL == 17) & port(b, 443)),
    ("UDP/123 (NTP)", (b.PROTOCOL == 17) & port(b, 123)),
    ("UDP/5353 (mDNS)", (b.PROTOCOL == 17) & port(b, 5353)),
    ("UDP/1900 (SSDP)", (b.PROTOCOL == 17) & port(b, 1900)),
    ("UDP/67-68 (DHCP)", (b.PROTOCOL == 17) & (port(b, 67) | port(b, 68))),
    ("ICMP (1)", b.PROTOCOL == 1),
    ("IGMP (2)", b.PROTOCOL == 2),
]
print("\nbenign coverage by traffic type (flows | share):")
for name, m in rows:
    print(f"  {name:22s}{int(m.sum()):8d} | {100 * m.mean():6.3f}%")
t = pd.to_datetime(df.FLOW_START_MILLISECONDS, unit="ms")
print("\ncapture span:", t.min(), "->", t.max(), "| distinct days:", t.dt.date.nunique())
print("unique benign destination ports:", b.L4_DST_PORT.nunique())
