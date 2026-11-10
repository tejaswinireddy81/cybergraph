# Feature Semantics Audit: NetFlow/nProbe v3 vs. Scapy FlowExtractor

Status: **Phase 1 Deliverable (P1). Audit complete.**
This document provides a feature-by-feature semantic comparison between the NF-UNSW-NB15-v3 dataset (generated via nProbe v3 NetFlow engine from pcaps) and the live Scapy-based `FlowExtractor`.

---

## 1. Classification Methodology

Each of the 47 model features used by CyberGraph is classified into one of three reliability tiers:
- **Tier A: Robust**: Header-derived or pure counting statistics whose physical semantics are identical across packet capture engines and NetFlow probes (e.g. byte counts, packet counts, TCP flags, port numbers).
- **Tier B: Timing/Rate-Sensitive**: Features that depend on packet timestamps. Highly reliable when computed from capture timestamps (`timestamp_source="capture"`), but susceptible to train/serve skew under wall-clock processing delays (`timestamp_source="wall"`).
- **Tier C: Exporter-Dependent / Fragile**: Features where nProbe and Scapy employ differing heuristics, application-layer parsers, or flow boundary definitions (e.g. L7 protocol classification, DNS fields, TCP retransmission tracking, and raw minimum packet length).

---

## 2. Feature-by-Feature Audit Table

| # | Feature Name | Tier | nProbe Semantics | Scapy `FlowExtractor` Semantics | Risk / Semantic Gap | Recommendation in F0–F5 |
|---|---|:---:|---|---|---|---|
| 1 | `L4_SRC_PORT` | A | Transport layer source port | Transport layer source port (`tcp.sport` / `udp.sport`) | Ephemeral client ports carry no generalizable attack signal | Dropped in F3, F4, F5; binned in F5 |
| 2 | `L4_DST_PORT` | A | Transport layer destination port | Transport layer destination port (`tcp.dport` / `udp.dport`) | memorizes service ports (e.g. 80, 443) | Dropped in F3, F4; binned in F5 |
| 3 | `PROTOCOL` | A | IPv4 protocol byte (6=TCP, 17=UDP, 1=ICMP) | IPv4 protocol byte (`ip.proto`) | Identical | Retained in F0-F3; one-hot in F4-F5 |
| 4 | `L7_PROTO` | C | Deep Packet Inspection via nDPI | Hardcoded lookup table `_PORT_TO_L7` mapping port to protocol | **Severe skew**: nDPI inspects application handshake; port table guesses by destination port | Dropped in F3, F4, F5 |
| 5 | `IN_BYTES` | A | Total bytes sent src→dst | Accumulated IPv4 packet length (`ip.len`) src→dst | Identical | Retained; log1p-transformed in F4-F5 |
| 6 | `IN_PKTS` | A | Total packets sent src→dst | Total IPv4 packets src→dst | Identical | Retained; log1p-transformed in F4-F5 |
| 7 | `OUT_BYTES` | A | Total bytes sent dst→src | Accumulated IPv4 packet length (`ip.len`) dst→src | Identical | Retained; log1p-transformed in F4-F5 |
| 8 | `OUT_PKTS` | A | Total packets sent dst→src | Total IPv4 packets dst→src | Identical | Retained; log1p-transformed in F4-F5 |
| 9 | `TCP_FLAGS` | A | Bitwise OR of all TCP flags across both directions | Bitwise OR of all observed TCP flags (`tcp_flags_all`) | Identical | Retained in all feature sets |
| 10 | `CLIENT_TCP_FLAGS` | A | Bitwise OR of TCP flags from flow initiator | Bitwise OR of TCP flags from forward direction | Identical when connection starts cleanly with SYN | Retained in all feature sets |
| 11 | `SERVER_TCP_FLAGS` | A | Bitwise OR of TCP flags from responder | Bitwise OR of TCP flags from reverse direction | Identical when connection starts cleanly with SYN | Retained in all feature sets |
| 12 | `FLOW_DURATION_MILLISECONDS` | B | `(last_pkt_time - first_pkt_time) * 1000` | In capture mode: identical. In wall mode: processing duration | Skewed if packet processing is delayed | Retained; exact under `capture` mode |
| 13 | `DURATION_IN` | B | Duration of forward stream (ms) | `(t_last_in - t_first_in) * 1000` | Identical under `capture` mode | Retained in all feature sets |
| 14 | `DURATION_OUT` | B | Duration of reverse stream (ms) | `(t_last_out - t_first_out) * 1000` | Identical under `capture` mode | Retained in all feature sets |
| 15 | `LONGEST_FLOW_PKT` | A | Maximum packet length in flow | `max(pkt_lengths)` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 16 | `SHORTEST_FLOW_PKT` | C | Minimum packet length in flow | `min(pkt_lengths)` with TCP floor | **Critical shortcut**: nProbe established flows >= 52B; raw Scapy ACKs = 40B | Dropped in F1, F2, F3, F4, F5 |
| 17 | `MIN_IP_PKT_LEN` | C | Minimum IP packet length in flow | `min(pkt_lengths)` with TCP floor | **Critical shortcut**: 79% XGBoost importance. Masks scans with floor | Dropped in F1, F2, F3, F4, F5 |
| 18 | `MAX_IP_PKT_LEN` | A | Maximum IP packet length in flow | `max(pkt_lengths)` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 19 | `SRC_TO_DST_SECOND_BYTES` | B/C | Average byte rate (bytes/s) src→dst | `in_bytes / dur_s` (clamped to 10 KB/s, 100ms floor) | Clamping mitigates single-packet rate explosion | Dropped in F4, F5 (replaced by log ratio) |
| 20 | `DST_TO_SRC_SECOND_BYTES` | B/C | Average byte rate (bytes/s) dst→src | `out_bytes / dur_s` (clamped to 5 KB/s, 100ms floor) | Clamping mitigates single-packet rate explosion | Dropped in F4, F5 (replaced by log ratio) |
| 21 | `RETRANSMITTED_IN_BYTES` | C | TCP retransmitted bytes src→dst via TCP state tracker | In-memory sequence number set check | Scapy sequence set reset on long flows; benign data has 97.5% retransmission | Dropped in F4, F5 |
| 22 | `RETRANSMITTED_IN_PKTS` | C | TCP retransmitted packets src→dst | Count of repeated sequence numbers seen | Heuristic match | Dropped in F4, F5 |
| 23 | `RETRANSMITTED_OUT_BYTES`| C | TCP retransmitted bytes dst→src | In-memory sequence number set check | Heuristic match | Dropped in F4, F5 |
| 24 | `RETRANSMITTED_OUT_PKTS` | C | TCP retransmitted packets dst→src | Count of repeated sequence numbers seen | Heuristic match | Dropped in F4, F5 |
| 25 | `SRC_TO_DST_AVG_THROUGHPUT`| B/C | Directional throughput (bps) | `(in_bytes * 8) / dur_in_s` (capped at 20 Mbps) | Clamped | Dropped in F4, F5 |
| 26 | `DST_TO_SRC_AVG_THROUGHPUT`| B/C | Directional throughput (bps) | `(out_bytes * 8) / dur_out_s` (capped at 30 Mbps) | Clamped | Dropped in F4, F5 |
| 27 | `NUM_PKTS_UP_TO_128_BYTES` | A | Packets with len <= 128 | Incremented if `len(ip) <= 128` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 28 | `NUM_PKTS_128_TO_256_BYTES` | A | Packets with 128 < len <= 256 | Incremented if `128 < len(ip) <= 256` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 29 | `NUM_PKTS_256_TO_512_BYTES` | A | Packets with 256 < len <= 512 | Incremented if `256 < len(ip) <= 512` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 30 | `NUM_PKTS_512_TO_1024_BYTES`| A | Packets with 512 < len <= 1024 | Incremented if `512 < len(ip) <= 1024` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 31 | `NUM_PKTS_1024_TO_1514_BYTES`| A | Packets with 1024 < len <= 1514 | Incremented if `len(ip) > 1024` | Identical | Retained in F0-F1; dropped in F2-F5 |
| 32 | `TCP_WIN_MAX_IN` | A | Maximum TCP window observed src→dst | `max(tcp_win)` forward direction | Identical | Retained in all feature sets |
| 33 | `TCP_WIN_MAX_OUT` | A | Maximum TCP window observed dst→src | `max(tcp_win)` reverse direction | Identical | Retained in all feature sets |
| 34 | `ICMP_TYPE` | A | ICMP type number | `icmp.type` | Identical | Dropped in F3, F4, F5 |
| 35 | `ICMP_IPV4_TYPE` | A | Duplicate of ICMP type | Duplicate of `icmp.type` | Identical | Dropped in F3, F4, F5 |
| 36 | `DNS_QUERY_ID` | C | First DNS transaction ID | Struct unpack of first 2 bytes on port 53 | Fragile; depends on unencrypted UDP/53 DNS | Dropped in F3, F4, F5 |
| 37 | `DNS_QUERY_TYPE` | C | First DNS QTYPE | Unpacked question record QTYPE | Fragile | Dropped in F3, F4, F5 |
| 38 | `DNS_TTL_ANSWER` | C | TTL of first DNS answer | Unpacked answer record TTL | Fragile | Dropped in F3, F4, F5 |
| 39 | `FTP_COMMAND_RET_CODE` | C | 3-digit FTP status code | ASCII parse of port 21 payload | Fragile; legacy protocol only | Dropped in F3, F4, F5 |
| 40 | `SRC_TO_DST_IAT_MIN` | B | Minimum forward inter-arrival time (ms) | `min(iat_in) * 1000` | Exact in capture mode | Retained in all feature sets |
| 41 | `SRC_TO_DST_IAT_MAX` | B | Maximum forward inter-arrival time (ms) | `max(iat_in) * 1000` | Exact in capture mode | Retained in all feature sets |
| 42 | `SRC_TO_DST_IAT_AVG` | B | Mean forward inter-arrival time (ms) | `mean(iat_in) * 1000` | Exact in capture mode | Retained in all feature sets |
| 43 | `SRC_TO_DST_IAT_STDDEV` | B | Stddev of forward IAT (ms) | `std(iat_in) * 1000` | Exact in capture mode | Retained in all feature sets |
| 44 | `DST_TO_SRC_IAT_MIN` | B | Minimum reverse inter-arrival time (ms) | `min(iat_out) * 1000` | Exact in capture mode | Retained in all feature sets |
| 45 | `DST_TO_SRC_IAT_MAX` | B | Maximum reverse inter-arrival time (ms) | `max(iat_out) * 1000` | Exact in capture mode | Retained in all feature sets |
| 46 | `DST_TO_SRC_IAT_AVG` | B | Mean reverse inter-arrival time (ms) | `mean(iat_out) * 1000` | Exact in capture mode | Retained in all feature sets |
| 47 | `DST_TO_SRC_IAT_STDDEV` | B | Stddev of reverse IAT (ms) | `std(iat_out) * 1000` | Exact in capture mode | Retained in all feature sets |

---

## 3. Summary of Findings & Feature Set Rationale

1. **The Primary Shortcut Trap (Tier C - `MIN_IP_PKT_LEN` / `SHORTEST_FLOW_PKT`)**:
   - In training data, benign flows are established (>= 52 bytes) and attack flows are often single SYN or bare ACK packets (40 bytes).
   - XGBoost split at 50 bytes accounts for ~80% of model weight.
   - Removing these two features produces **Feature Set F1** (45 features).
2. **The Secondary Shortcut Trap (Tier C - Packet Size Histogram)**:
   - Attack packets cluster tightly into specific size buckets (e.g. 40B SYN scans in `NUM_PKTS_UP_TO_128_BYTES`).
   - Removing all packet length and histogram features produces **Feature Set F2** (38 features).
3. **Generator & Identity Artifacts (Tier C - Port numbers, DNS, FTP, L7, ICMP)**:
   - UNSW benign traffic exhibits generator artifacts (42,798 destination ports, port-based L7 guessing).
   - Removing identity and protocol-specific fields produces **Feature Set F3** (29 features).
4. **Behavioral & Scale-Invariant Modeling (Feature Set F4 - 32 features)**:
   - Replaces raw byte rates with scale-invariant ratios (`bytes_per_packet_in`, `bytes_per_packet_out`, `log_byte_ratio`, `log1p(dur)`).
   - Immune to rate-clipping and flow-duration jitter.
5. **Port-Binned Generalization (Feature Set F5 - 34 features)**:
   - Categorizes ports into System (<1024), Registered (1024-49151), and Ephemeral (>=49152), providing context without memorizing exact port numbers.
