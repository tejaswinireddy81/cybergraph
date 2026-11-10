# Flow extractor: investigation and implementation proposal

Status: **investigation and proposal only.** The production `flow_extractor.py` is **not changed** by this work (the one earlier
addition, the `min_pkt_floor_tcp` constructor argument, keeps the default of 52). The fixes exist as a separate **prototype**
in `proposals/extractor/` that nothing imports. Nothing here retrains a model, touches a production model file, or changes the
firewall (still dry-run). Nothing is committed.

Decision received: the two fixes are approved *as a separate reviewed change*, kept separate from model retraining, and
production behaviour must stay as it is until the change passes regression testing. This document is that review package.

## 1. Summary

| # | Finding | Evidence | Proposal |
|---|---|---|---|
| F1 | A TCP flow is exported at the **first FIN/RST**; the server's FIN-ACK and the client's final ACK then start **orphan one-packet flows** | synthetic: 3.00 flows per connection, 66.7 % single-packet. Real earlier capture: 24 % of TCP flows (21/88) single-packet; **training data: 0.00 % (10/283,495)** | `tcp_close_mode="linger"` |
| F2 | All timing features use `time.time()` **at processing**, not the packet's capture time | measured extractor cost 529 µs/packet; at that cost UDP-flood flows are 93 % wrong in duration and 48 % in throughput | `timestamp_source="capture"` |
| F3 | `add_packet` (sniff thread) and `flush_expired` (flush thread) mutate the same dict **without a lock** | `KeyError` reproduced: 4 errors in 4 short stress runs vs 0 with a lock | an `RLock` |
| F4 | Two existing patches **mask** attacks and are a model-stage decision, **not part of this change**: the 52-byte TCP floor and the byte-rate caps | floor ON: SYN-scan recall 21 %; caps clip floods | left unchanged |

The prototype with default flags is **bit-identical** to production (all 54 compared columns, 4,950 flows), so merging the
flags changes nothing until a flag is switched.

## 2. F1: orphan flows from first-FIN export

**Mechanism** (`flow_extractor.py:562-566`): when any packet has FIN or RST set, the flow is exported and deleted. A normal teardown
is client FIN-ACK, server FIN-ACK, client ACK. After the first FIN-ACK the flow is gone, so:
1. the server's FIN-ACK starts a new flow (server as "initiator") that *itself* contains a FIN, so it is exported at once: 1 packet;
2. the client's final ACK starts another flow that sits until the 10 s idle timeout: 1 packet.
One connection therefore becomes **three** records. Windows clients also send RST/RST-ACK after FIN, producing more.

**Evidence**

| | flows per connection | single-packet | bidirectional | both FINs | packets p5 / p50 / p95 |
|---|---|---|---|---|---|
| Training data (UNSW benign TCP, 190,907 flows) | n/a | **0.0 %** | **100 %** | **74.5 %** | 8 / 36 / 248 |
| Production extractor, synthetic (273 connections) | **3.00** | **66.7 %** | 33.3 % | 0 % | 1 / 1 / 39 |
| Prototype `linger`, same packets | **1.00** | 0.0 % | 100 % | 100 % | 8 / 28 / 52 |

* Real earlier capture (this machine): 21 of 88 TCP flows were single-packet (flags 0x11 FIN-ACK from the server and 0x14 RST-ACK
  from the client); training data: 10 of 283,495.
* The training data has **no RST in benign TCP flows at all** (0 of 190,907) and 0.48 % in attack flows.
* Because of the 52-byte floor these orphans score ~0.01 today. With their true 40-byte wire length the production model flags
  **14 of 21**: the floor is hiding this artifact (as well as the SYN-scan blind spot).
* Caveat: the synthetic flows are shorter than real ones (p95 52 vs 248 packets). Flow shape, not length, is what this shows.

**Training-data semantics to match:** a NetFlow record keeps the whole connection, including both FINs (74.5 %) and is always
bidirectional. Exporting at the first FIN cannot produce that.

## 3. F2: processing timestamps instead of capture timestamps

**Mechanism:** `_process` starts with `now = time.time()` (`flow_extractor.py:353`) and uses it for `t_first`, `t_last`, every IAT,
`DURATION_IN/OUT`, the flow duration and everything derived from them (rates, throughput, `_timestamp`). Scapy sets the capture
timestamp on every sniffed packet (`pkt.time = ts` from the libpcap header; `scapy/supersocket.py:219` and `:389`, read from
source; **not exercised live**, because capture stays disabled). The training data was exported by nProbe from pcaps, i.e. with
capture timestamps.

**Measured:** `add_packet` costs **529 µs per packet** on this machine (best of 3, offline, 20,000 packets). It cannot keep up
with more than about 1,900 packets/s on one core; faster bursts are silently stretched in time.

**Effect on the production (wall-clock) extractor, synthetic traffic, truth = capture timestamps** (per-class medians; the 5x and
20x rows stand in for queueing behind a busy thread or Npcap batch delivery, **assumed, not measured**):

| class | at measured cost (1x) | 5x | 20x |
|---|---|---|---|
| UDP flood: duration error median (p90) / flows >10 % off | **93 % (107 %) / 100 %** | 866 % / 100 % | 3,766 % / 100 % |
| UDP flood: throughput error median | 48 % | 90 % | 97 % |
| DNS: flows >10 % off in duration | 0 % | 45 % | 96 % |
| QUIC: flows >10 % off | 0 % | 0 % | 17 % |
| HTTPS, scans, SYN flood (1-packet flows), ICMP sweep | ~0 % | ~0-1 % | ~0-4 % |

Single-packet flows are unaffected (no duration), which is why floods and scans of one packet look fine. Many-packet bursts are
where the signal is distorted, and those carry the attack evidence. **The capture-timestamp extractor is exact under every delay
(asserted by a test).**

## 4. F3: thread safety

`live_capture.py` runs `add_packet` on the sniff thread and `flush_expired` on `FlowFlushThread` every 2 s, with no lock around the
shared `_flows` dict. Stress test (idle timeout 0 to force overlap, 4 runs x 3 s): production **4 `add_packet` errors**
(`KeyError` on a flow key already deleted by the flusher, logged and swallowed), prototype with lock **0**. With the real 10 s idle
timeout the window is narrow, so in practice this is a rare loss/corruption of a flow rather than a frequent failure; the lock is
cheap, and the proposed linger logic touches the same structures from both threads, so it is needed anyway.

## 5. F4: what is deliberately NOT changed here

These interact with the model and belong to the model stage; they are listed so the review is complete.
* **TCP 52-byte floor.** After the orphan fix, with the floor OFF: flows from Unix-style clients are 0.0 % flagged, but flows from a
  **Windows-style client (bare 40-byte ACKs) are 98.9 % flagged**. So fixing the extractor alone does not remove the model's
  dependence on `MIN_IP_PKT_LEN`; that needs the retrained feature sets (F1-F5), where the floor becomes dead code.
* **Byte-rate caps and the 100 ms minimum denominator** (10 KB/s etc.): they clip flood rates into the training range. With
  accurate timestamps they would still hide high-rate attacks; revisit together with F4/F5.
* **Other train/serve differences noticed** (not fixed, need a decision): direction is "first packet seen" (a capture that starts
  mid-connection flips it); long flows split at the 60 s active timeout; `L7_PROTO` comes from a port table, not nDPI;
  retransmissions count repeated sequence numbers (the training data does too: 97.5 % of benign TCP flows have some, so this is
  left as is). A diagnostic `_midflow` flag is added so partial flows can be identified.

## 6. The proposal

All changes are flags on `FlowExtractor.__init__`; **defaults equal today's behaviour**.

| Flag | Default | Proposed final value | What it does |
|---|---|---|---|
| `timestamp_source` | `"wall"` | `"capture"` | Features use `pkt.time`. Missing/invalid timestamps fall back to the wall clock for that packet (counted in `capture_ts_fallbacks`). Out-of-order timestamps never move a flow backwards. **Expiry always uses wall-clock arrival** (`wall_first/wall_last`), exactly as today, so live capture and replay of old captures both keep working. |
| `tcp_close_mode` | `"legacy"` | `"linger"` | See below. |
| `tcp_linger_s` | 2.0 | 2.0 (to tune) | quiet time after FIN/RST before a flow is exported |
| lock | always on | | `RLock` around `add_packet`, `flush_expired`, `active_flow_count` |

**`linger` semantics**
* FIN/RST **marks** the flow as closing instead of exporting it; later packets of the connection join the same record.
* **Clean teardown** (a FIN from both sides, then the first closer's ACK): exported **immediately** from `add_packet`, so no latency is added in the common case.
* Anything else (RST, half-closed, FIN never answered): exported by `flush_expired` after `tcp_linger_s` of quiet.
* A new SYN on a closing 5-tuple (port reuse) first finishes the old connection, then starts a new one.
* After export a short **tombstone** (2 s) swallows stray late packets of that connection so they cannot become orphans (a fresh SYN is never swallowed). Swallowed packets are counted in `stray_packets_swallowed`.
* New diagnostics fields (not model inputs): `_close_reason` (`fin_complete`, `linger`, `port_reuse`, `idle_timeout`, `active_timeout`, `drain`, `legacy_fin_rst`) and `_midflow`.
* `drain_all()` finalises everything (shutdown, offline replay, tests).

**API compatibility:** `add_packet` still returns `Optional[dict]` and `flush_expired` still returns a list. `live_capture.py` already
processes flows from both paths, so **no change to `live_capture.py` or the models' input columns** is needed. Latency:
clean teardowns unchanged; RST/half-closed flows now appear up to `tcp_linger_s` + the 2 s flush interval later (about 4 s worst case).

## 7. Effect on the model (E3: production XGBoost F0, alert at 0.75, **synthetic, provisional**)

| configuration | benign FPR | benign TCP FPR | TCP FPR Windows-style | TCP FPR Unix-style | non-TCP FPR | recall all | BruteForce | DoS | Scan |
|---|---|---|---|---|---|---|---|---|---|
| production, floor ON (today) | 19.1 % | 0.0 % | 0.0 % | 0.0 % | 58.4 % | 13.9 % | 0.0 % | 1.1 % | 21.6 % |
| production, floor OFF | 34.0 % | 22.1 % | 66.3 % | 0.0 % | 58.4 % | 88.5 % | 20.7 % | 100 % | 100 % |
| **proposed**, floor ON | 34.7 % | 0.0 % | 0.0 % | 0.0 % | 58.4 % | 15.4 % | 0.0 % | 1.1 % | 21.6 % |
| **proposed**, floor OFF | 48.1 % | 33.0 % | 98.9 % | 0.0 % | 58.4 % | 96.3 % | 31.1 % | 100 % | 100 % |

Reading it:
* The extractor fix does **not** by itself improve the model's behaviour; it removes artifacts and makes the inputs
  comparable with the training data. Non-TCP results are identical.
* The *overall* benign FPR goes up with the fix only because the orphan flows (which scored 0 % with the floor) disappear from
  the denominator. Compare per class, not overall.
* With the floor OFF the model's shortcut is exposed (Windows-style ACK size). That is what the F1-F5 study addresses.
* Neither fix is a reason to retrain by itself, and the fix must not be judged by this table; it is judged by the regression criteria below.

## 8. Regression testing and acceptance criteria

Built and passing now (23 tests in `tests/test_extractor_proposal.py`, plus the existing 178 unchanged):
* default-flag prototype == production, bit for bit, on mixed synthetic traffic (TCP, UDP, QUIC, IGMP, scan, flood, brute force);
* production is **guarded** against accidental change: its constructor signature and source must stay as they are until approval;
* linger: one flow per connection with both FINs; clean teardown exported immediately; stragglers swallowed; fresh SYN not swallowed; half-closed exported by the timer; RST handling; port reuse; mid-connection start flagged; UDP/ICMP/idle expiry unchanged;
* capture timestamps: exact durations/IATs under 500 ms of processing delay; wall mode still measures processing time; missing/negative/NaN/inf/garbage timestamps fall back; out-of-order timestamps give no negative intervals; replaying a 1985 capture still expires flows;
* timing features independent of processing delay on synthetic traffic; thread-safety with concurrent capture and flush; model feature columns unchanged and finite.

**Proposed gates before any production default is flipped**
1. Full existing suite green; bandit and the flake8 correctness gate green.
2. Default-flag equivalence test stays green (merging the flags changes nothing).
3. On synthetic traffic: orphan share 0 %; flows per connection 1.0; capture-timestamp features exact under the delay model.
4. Non-TCP flow features identical between modes when there is no processing delay.
5. **Shadow run on live traffic** (both modes computed side by side, only the old one used): orphan share, `_close_reason` mix, latency, memory, `capture_ts_fallbacks`. **Needs a live capture, so it waits for your capture sign-off.**
6. Only then: flip `timestamp_source` and `tcp_close_mode` defaults, in a change separate from any model change.

## 9. Rollout and rollback

| Stage | Change | Behaviour change |
|---|---|---|
| S0 | merge the flags (defaults legacy) + lock + metadata | **none** (equivalence test) |
| S1 | evaluation tooling (synthetic now, lab later) uses `capture` + `linger` | none in production |
| S2 | live shadow mode after lab sign-off | none (observation only) |
| S3 | flip defaults, **separate from model retraining** | yes: complete TCP flows, accurate timing |
Rollback at any stage: set the two flags back to `legacy`/`wall`.

## 10. Risks and unknowns
* The 2 s linger and 2 s tombstone are reasonable defaults, **not validated on real traffic**.
* Capture timestamp behaviour was confirmed by reading scapy's source, **not by a live capture**; the fallback path covers a surprise.
* The synthetic processing-delay multipliers (5x, 20x) are assumptions; only the 529 µs/packet cost is measured.
* The tombstone table is cleaned by `flush_expired`; if the flush thread stopped, it would grow (bounded only by connection rate). A hard size cap can be added in the real change.
* Direction flips for mid-connection captures, long-flow splitting and the L7 table are unchanged and remain train/serve differences.
* Synthetic traffic is a model of traffic; real flows are longer and messier.

## 11. Reproduce
```
python scripts/experiments/extractor_proposal_eval.py      # E0-E4 above (about 5 minutes)
python scripts/experiments/extractor_processing_cost.py    # measured cost + per-class timestamp error (about 90 s)
python -m pytest tests/test_extractor_proposal.py -q
git diff --no-index flow_extractor.py proposals/extractor/flow_extractor_proposed.py   # the review diff (164 added, 21 removed lines)
```
Files: `proposals/extractor/flow_extractor_proposed.py` (prototype), `proposals/extractor/flow_extractor.patch` (diff).

## 12. Decisions needed
1. Approve the flag design (section 6), the defaults, and the linger values (or choose others).
2. Approve implementing S0 in production `flow_extractor.py` (flags default to today's behaviour; the equivalence test proves it).
3. Confirm that S2/S3 wait for the lab capture sign-off (a live shadow run needs a capture).
4. Decide whether the thread-safety lock goes into S0 (recommended: yes, it changes no output).
