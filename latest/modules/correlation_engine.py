"""
modules/correlation_engine.py
==============================
Rule-Based Temporal Correlation Layer

Tracks per-source-IP event timelines and detects escalating attack patterns
by correlating GNN risk scores and attack type sequences over time.

Architecture note:
  This is a RULE-BASED layer, not an additional ML model.
  It applies temporal aggregation and deterministic stage-mapping to the
  GNN-scored events produced upstream by the Dynamic Bipartite Temporal GNN.

Observable Kill Chain stages (NetFlow-derivable only):
  RECONNAISSANCE, DELIVERY, EXPLOITATION, COMMAND_AND_CONTROL, ACTION
  Weaponization is omitted — it occurs before delivery and is not observable
  from NetFlow-style traffic records.

Correlation score formula (exponential temporal decay):
  C_ip = (1/N) * sum r_i * 0.85^(N-1-i)
  where r_i is the GNN edge-risk of event i.
"""

import time
from collections import defaultdict, deque

STAGE_RANK = {
    "RECONNAISSANCE":      1,
    "DELIVERY":            2,
    "EXPLOITATION":        3,
    "COMMAND_AND_CONTROL": 4,
    "ACTION":              5,
}

_TYPE_TO_STAGE = {
    "Reconnaissance": "RECONNAISSANCE",
    "Normal":         None,
    "Fuzzer":         "DELIVERY",
    "Generic":        "DELIVERY",
    "Exploits":       "EXPLOITATION",
    "Backdoor":       "COMMAND_AND_CONTROL",
    "DoS":            "ACTION",
}

DECAY_LAMBDA          = 0.85
MAX_EVENTS_PER_IP     = 50
ESCALATION_WINDOW     = 4
CORRELATION_THRESHOLD = 0.40


class CorrelationEngine:
    """
    Rule-based temporal correlation layer.

    Maintains a per-source-IP rolling event timeline (up to 50 events).
    Computes an exponential-decay correlation score and maps the observed
    attack type sequence to the highest observable Kill Chain stage.
    """

    def __init__(self):
        self._timelines = defaultdict(lambda: deque(maxlen=MAX_EVENTS_PER_IP))

    def record_event(self, *, src_ip, dst_ip, dst_port, protocol, attack_type,
                     gnn_risk, xgb_prob, ae_score, fusion_score, risk_level,
                     geo, capture_ms, feature_prep_ms, gnn_infer_ms):
        """Record one detection event. All values must be actual measurements."""
        stage = _TYPE_TO_STAGE.get(attack_type)
        event = {
            "timestamp":    time.strftime("%H:%M:%S"),
            "src_ip":       str(src_ip),
            "dst_ip":       str(dst_ip),
            "dst_port":     int(dst_port),
            "protocol":     int(protocol),
            "attack_type":  str(attack_type),
            "gnn_risk":     round(float(gnn_risk),    4),
            "xgb_prob":     round(float(xgb_prob),    4),
            "ae_score":     round(float(ae_score),     4),
            "fusion_score": round(float(fusion_score), 4),
            "risk_level":   str(risk_level),
            "stage":        stage,
            "geo":          geo if geo else {},
            "latency": {
                "capture_ms":      round(float(capture_ms),      2),
                "feature_prep_ms": round(float(feature_prep_ms), 2),
                "gnn_infer_ms":    round(float(gnn_infer_ms),    2),
                "total_ms":        round(float(capture_ms)+float(feature_prep_ms)+float(gnn_infer_ms), 2),
            },
        }
        if stage is not None:
            self._timelines[src_ip].append(event)
        return self._build_report(src_ip)

    def get_top_correlated(self, top_n=5):
        reports = [self._build_report(ip) for ip in list(self._timelines)]
        reports = [r for r in reports if r["correlation_score"] >= CORRELATION_THRESHOLD]
        reports.sort(key=lambda x: x["correlation_score"], reverse=True)
        return reports[:top_n]

    def get_timeline(self, src_ip):
        return list(self._timelines.get(src_ip, []))

    def clear(self):
        self._timelines.clear()

    def _build_report(self, src_ip):
        events = list(self._timelines.get(src_ip, []))
        n = len(events)
        if n == 0:
            return _empty_report(src_ip)
        score = sum(ev["gnn_risk"] * (DECAY_LAMBDA ** (n - 1 - i))
                    for i, ev in enumerate(events))
        correlation_score = round(score / n, 4)
        highest_stage, highest_rank = None, 0
        for ev in events:
            s = ev.get("stage")
            if s and STAGE_RANK.get(s, 0) > highest_rank:
                highest_stage = s
                highest_rank  = STAGE_RANK[s]
        if n >= ESCALATION_WINDOW:
            recent = [ev["gnn_risk"] for ev in events[-ESCALATION_WINDOW:]]
            escalation = all(recent[i] <= recent[i+1] for i in range(len(recent)-1))
        else:
            escalation = False
        return {
            "src_ip":              src_ip,
            "event_count":         n,
            "correlation_score":   correlation_score,
            "kill_chain_stage":    highest_stage or "NONE",
            "escalation_detected": escalation,
            "attack_sequence":     [ev["attack_type"] for ev in events[-10:]],
            "risk_trend":          [ev["gnn_risk"]    for ev in events[-10:]],
            "geo":                 events[-1].get("geo", {}),
            "timeline":            events[-8:],
        }


def _empty_report(src_ip):
    return {"src_ip": src_ip, "event_count": 0, "correlation_score": 0.0,
            "kill_chain_stage": "NONE", "escalation_detected": False,
            "attack_sequence": [], "risk_trend": [], "geo": {}, "timeline": []}
