# Lab capture: requirements to confirm BEFORE any capture

Status: **no capture has been started and none will be until every item in section 7 is signed off.**
Until then the study uses existing datasets and synthetic traffic only (`evaluation/synthetic_traffic.py`).

Captures go to `latest/data/lab_captures/` (gitignored, may contain real hostnames/IPs and DNS names, never committed).

## 1. Permission and scope (needed from you)

* Confirm the network is yours (or you are authorised to capture on it) and who else uses it.
* Confirm that people on that network accept their traffic being captured, or that only your own devices are captured.
* Confirm that generating scan/flood traffic is permitted **on the isolated segment only** (section 2).

## 2. Isolation

* **Benign capture:** your normal network is acceptable if only authorised devices are captured and DNS/URL content is
  handled as in section 6.
* **Attack generation must be isolated:** a separate switch/router, VLAN or VM host-only network with **no route to the
  internet or to other people's devices**. Attacker and target devices both belong to you.
* The ML dashboard's firewall stays in **dry-run**; no experiment calls the block API; no real firewall rule is created.

## 3. Devices and roles (please fill in)

| Role | What | OS | Owner | Notes |
|---|---|---|---|---|
| Capture host | machine running the capture (needs Npcap/libpcap and admin rights) | | | |
| Benign client(s) (>= 2 devices) | generate normal traffic | | | |
| Server/target(s) | targets for attacks (VMs recommended) | | | |
| Attacker host | runs the attack tools | | | isolated segment only |
| Capture point | host NIC / SPAN-mirror port / tap | | | which interface |

Why >= 2 benign devices and >= 5 sessions over >= 3 days: so train/validation/test can be split by device/session/day
and no related flows straddle two splits.

## 4. Benign traffic plan (proposed minimum flows)

DNS (UDP+TCP, several resolvers incl. 8.8.4.4) 5,000; HTTPS/TLS 10,000; QUIC/HTTP3 2,000; other TCP (HTTP, SSH, SMB, IMAP)
5,000; other UDP (NTP, DHCP, mDNS, SSDP) 2,000; ICMP 500; IGMP/multicast 300 (generated deliberately, rare); background
desktop traffic for the remainder. ARP is captured as well but is **out of scope for the flow model** (handled by the
separate optional ARP module).

Benign activity is scripted and known; flows that cannot be vouched for (unexpected scanning, traffic to known-bad addresses
from `datasets/threat_intelligence.json`) are excluded, not guessed.

## 5. Attack plan (needs your approval of each tool and its limits)

| Attack | Typical tool | Proposed limit |
|---|---|---|
| SYN scan (default and bare SYN) | nmap, hping3 | one /24 target range, rate-limited |
| Port sweep | nmap, masscan | single port across the range |
| SYN flood / UDP flood | hping3 | short bursts, capped packet rate, isolated segment |
| ICMP sweep | nmap -sn | the range |
| Brute-force-like auth failures | scripted login attempts | against your own test VM |

State which tools you accept and the rate caps. The **>= 90 % recall target** is measured on these, reported **per
attack family** (Scan, DoS, BruteForce) and never only as an average.

## 6. Data handling and privacy

* Raw packet payloads are not needed: store flow records (the extractor's feature rows) plus metadata. If pcaps are kept,
  they stay local and are deleted after feature extraction unless you decide otherwise.
* Hostnames, DNS names, external IPs and MAC addresses are pseudonymised in anything that leaves the capture folder
  (tables, plots, reports).
* Retention: you choose (proposed: raw pcaps deleted after extraction, flow tables kept while the study runs).
* Each session gets an id, a device id, a date and a SHA-256 of its flow table, written to a manifest.

## 7. Sign-off checklist (nothing starts until all are ticked)

| # | Item | Confirmed by you |
|---|---|---|
| 1 | I own / am authorised to capture on this network | |
| 2 | Other users' traffic is excluded or they agreed | |
| 3 | An isolated segment exists for attacks (no internet route) | |
| 4 | Device list in section 3 is filled in | |
| 5 | Capture point and interface chosen | |
| 6 | Attack tools and rate limits in section 5 approved | |
| 7 | Privacy/retention choices in section 6 approved | |
| 8 | Firewall stays dry-run during the experiments | |
| 9 | Go-ahead to build and run the capture tooling | |

## 8. What I will and will not do

* Will: prepare the capture/labelling tooling only **after** sign-off; keep everything under `data/lab_captures/`;
  record a manifest per session; keep the captures out of Git.
* Will not: capture anything, scan anything, send any packet, or touch any firewall rule before sign-off.
* Synthetic traffic is built in memory and is never sent on a network (a test enforces this).
