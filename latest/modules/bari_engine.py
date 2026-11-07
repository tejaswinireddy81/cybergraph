"""
modules/bari_engine.py
=======================
Behavioral Attack-Route Intelligence (BARI) — Temporal Route Intelligence Engine

Learns host -> service transition patterns across temporal network snapshots rather than
relying solely on individual destination-port frequencies.

Components & Route Memory Architecture:
  1. Benign Route Memory:
     - Single-service access probability P(S_t | benign)
     - Sequential Markov transition probability matrix P(S_t | S_{t-1}, benign)
     - Data-Driven Service Sensitivity S(service) derived from benign access frequency & entropy
  2. Attacker Route Memory:
     - Per-host stateful transition sequence tracking [S_0, S_1, S_2, ...]
     - Inter-arrival velocity & trajectory tracking
  3. 6 BARI Component Sub-Metrics:
     - Route Novelty (RN)        : 1 - P(S_t | S_{t-1}, benign)
     - Route Deviation (RD)      : 1 - P(S_t | host_history)
     - Attack-Surface Progression (ASP) : Delta toward higher sensitivity services
     - Route Acceleration (RA)   : Inter-arrival transition velocity between distinct services
     - Route Persistence (RP)    : Accumulated non-benign transition count across snapshot window
     - Temporal GNN Risk (GNN)   : Structural & temporal GNN risk score
"""

import time
import math
from collections import defaultdict, deque

# Default fallback sensitivities for common service ports (used before baseline learning or fallback)
DEFAULT_SERVICE_SENSITIVITY = {
    80: 0.10, 443: 0.10, 8080: 0.15, 8443: 0.15, 8000: 0.15,
    53: 0.20, 25: 0.30, 110: 0.25, 143: 0.25, 465: 0.30, 587: 0.30,
    21: 0.50, 23: 0.60, 22: 0.65, 3306: 0.75, 5432: 0.75, 1433: 0.80, 1521: 0.80,
    3389: 0.85, 5900: 0.80, 445: 0.85, 139: 0.75, 137: 0.70, 161: 0.60,
    4444: 1.00, 4445: 1.00, 5555: 1.00, 6666: 0.95, 7777: 0.90, 31337: 1.00,
}
FALLBACK_SENSITIVITY = 0.45
PROGRESSION_WINDOW = 15

# Sub-metric fusion weights (documented hyperparameters)
W_RN  = 0.20  # Route Novelty
W_RD  = 0.15  # Route Deviation
W_ASP = 0.20  # Attack-Surface Progression
W_RA  = 0.15  # Route Acceleration
W_RP  = 0.10  # Route Persistence
W_GNN = 0.20  # Temporal GNN Risk


def _service_key(port: int, protocol: int = 6) -> str:
    return f"{int(port)}/{int(protocol)}"


def _port_label(port: int) -> str:
    port_names = {
        80: "HTTP", 443: "HTTPS", 53: "DNS", 22: "SSH", 21: "FTP", 23: "Telnet",
        25: "SMTP", 3306: "MySQL", 5432: "PostgreSQL", 1433: "MSSQL",
        3389: "RDP", 445: "SMB", 4444: "Backdoor/C2", 5555: "C2", 31337: "EliteBackdoor"
    }
    name = port_names.get(int(port), "SVC")
    return f"{name}({port})"


class BARIEngine:
    """
    Behavioral Attack-Route Intelligence (BARI) Engine with Route Memory.
    """

    def __init__(self):
        # Benign Route Memory
        self._benign_service_counts: dict = defaultdict(int)
        self._benign_transition_counts: dict = defaultdict(lambda: defaultdict(int))
        self._benign_total_flows: int = 0
        self._learned_sensitivity: dict = {}
        self._baseline_learned: bool = False

        # Attacker Route Memory: per host state
        # Stores list of dicts: {"service_key": str, "port": int, "sensitivity": float, "time": float}
        self._attacker_route_memory: dict = defaultdict(lambda: deque(maxlen=PROGRESSION_WINDOW))

    def learn_normal_baseline(self, df_benign):
        """
        Build Benign Route Memory from benign (Label=0) training flows.
        Derives data-driven service sensitivities and Markov transition probabilities.
        """
        if df_benign is None or len(df_benign) == 0:
            return

        port_col = "L4_DST_PORT" if "L4_DST_PORT" in df_benign.columns else "dst_port"
        proto_col = "PROTOCOL" if "PROTOCOL" in df_benign.columns else "protocol"
        src_col = "IPV4_SRC_ADDR" if "IPV4_SRC_ADDR" in df_benign.columns else "src_ip"

        counts = defaultdict(int)
        transitions = defaultdict(lambda: defaultdict(int))
        host_last_service = {}
        total = 0

        for _, row in df_benign.iterrows():
            port = int(row.get(port_col, 0))
            proto = int(row.get(proto_col, 6))
            s_key = _service_key(port, proto)
            src = str(row.get(src_col, "0.0.0.0"))

            counts[s_key] += 1
            total += 1

            if src in host_last_service:
                prev_s = host_last_service[src]
                transitions[prev_s][s_key] += 1
            host_last_service[src] = s_key

        self._benign_service_counts = counts
        self._benign_transition_counts = transitions
        self._benign_total_flows = max(total, 1)

        # Compute Data-Driven Service Sensitivity S(service)
        # Higher access frequency in benign corpus -> lower sensitivity score
        max_count = max(counts.values()) if counts else 1
        learned_sens = {}
        for s_key, cnt in counts.items():
            freq = cnt / total
            # Data-driven formula: S = clip(1.0 - sqrt(freq * 5.0), 0.05, 1.0)
            sens = max(0.05, min(1.0, 1.0 - math.sqrt(freq * 5.0)))
            learned_sens[s_key] = round(sens, 4)

        self._learned_sensitivity = learned_sens
        self._baseline_learned = True

    def get_service_sensitivity(self, dst_port: int, protocol: int = 6) -> float:
        """Data-driven service sensitivity score in [0.05, 1.00]."""
        s_key = _service_key(dst_port, protocol)
        if s_key in self._learned_sensitivity:
            return self._learned_sensitivity[s_key]
        return DEFAULT_SERVICE_SENSITIVITY.get(int(dst_port), FALLBACK_SENSITIVITY)

    def predict_next_targets(self, current_port: int, protocol: int = 6, top_k: int = 4) -> list:
        """
        Forecasts top_k likely next target services based on Markov transition probabilities
        and service sensitivity progression.
        Returns list of dicts: [{"port": int, "label": str, "probability": float, "sensitivity": float}, ...]
        """
        COMMON_TRANSITIONS = {
            80:   [(8080, 0.45), (22, 0.30), (3306, 0.15), (443, 0.10)],
            443:  [(80, 0.40), (8443, 0.30), (22, 0.20), (3306, 0.10)],
            8080: [(3306, 0.45), (5432, 0.25), (22, 0.20), (3389, 0.10)],
            22:   [(3389, 0.40), (3306, 0.30), (445, 0.20), (1433, 0.10)],
            21:   [(22, 0.45), (80, 0.30), (3306, 0.15), (4444, 0.10)],
            3306: [(5432, 0.35), (3389, 0.30), (4444, 0.25), (1433, 0.10)],
            3389: [(445, 0.50), (4444, 0.35), (3306, 0.10), (22, 0.05)],
            445:  [(3389, 0.45), (4444, 0.35), (139, 0.15), (31337, 0.05)],
        }
        
        curr_key = _service_key(current_port, protocol)
        trans_dict = self._benign_transition_counts.get(curr_key, {})
        tot = sum(trans_dict.values())
        
        candidates = []
        if tot > 5:
            for s_key, cnt in trans_dict.items():
                p = cnt / tot
                try:
                    port_val = int(s_key.split("/")[0])
                except Exception:
                    port_val = 80
                candidates.append((port_val, p))
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates = COMMON_TRANSITIONS.get(int(current_port), [(3306, 0.45), (22, 0.30), (3389, 0.15), (4444, 0.10)])

        results = []
        sum_p = sum(c[1] for c in candidates[:top_k])
        for p_val, prob in candidates[:top_k]:
            norm_prob = round(prob / max(sum_p, 1e-5), 4)
            sens = self.get_service_sensitivity(p_val)
            results.append({
                "port": int(p_val),
                "label": _port_label(p_val),
                "probability": float(norm_prob),
                "sensitivity": float(sens),
            })
        return results


    def score(self, *, src_ip: str, dst_port: int, protocol: int = 6,
              gnn_risk: float = 0.0, timestamp: float = None,
              use_memory: bool = True, use_progression: bool = True) -> dict:
        """
        Computes stateful BARI score and 6 route sub-metrics for a flow.
        """
        now = timestamp if timestamp is not None else time.time()
        curr_key = _service_key(dst_port, protocol)
        curr_sens = self.get_service_sensitivity(dst_port, protocol)

        history = self._attacker_route_memory[src_ip]
        prev_entry = history[-1] if history else None
        prev_key = prev_entry["service_key"] if prev_entry else None

        # 1. Route Novelty (RN): Transition novelty vs Benign Route Memory
        rn = self._compute_route_novelty(prev_key, curr_key)

        # 2. Route Deviation (RD): Deviation from host's own previous route history
        rd = self._compute_route_deviation(history, curr_key) if use_memory else rn

        # 3. Attack-Surface Progression (ASP): Movement toward higher sensitivity
        asp = self._compute_surface_progression(history, curr_sens) if use_progression else curr_sens

        # 4. Route Acceleration (RA): Inter-arrival transition velocity
        ra = self._compute_route_acceleration(history, now)

        # 5. Route Persistence (RP): Accumulated non-benign transition count
        rp = self._compute_route_persistence(history, curr_sens, rn)

        # 6. Temporal GNN Risk (GNN)
        gnn_score = max(0.0, min(1.0, float(gnn_risk)))

        # Final BARI Fusion Score
        if use_memory and use_progression:
            bari_score = (W_RN * rn + W_RD * rd + W_ASP * asp +
                          W_RA * ra + W_RP * rp + W_GNN * gnn_score)
        elif not use_memory:
            # Without memory ablation: only current flow signals
            bari_score = 0.35 * rn + 0.35 * asp + 0.30 * gnn_score
        else:
            # Without progression ablation
            bari_score = 0.30 * rn + 0.30 * rd + 0.40 * gnn_score

        bari_score = max(0.0, min(1.0, float(bari_score)))

        # Append to Attacker Route Memory
        history.append({
            "service_key": curr_key,
            "port": int(dst_port),
            "sensitivity": curr_sens,
            "time": now,
            "label": _port_label(dst_port),
        })

        # Format observed route string for XAI
        observed_route = [item["label"] for item in history]
        route_str = " -> ".join(observed_route)

        explanation_lines = []
        if rn > 0.60:
            explanation_lines.append(f"Route Novelty HIGH ({rn:.2f})")
        if rd > 0.60:
            explanation_lines.append(f"Route Deviation HIGH ({rd:.2f})")
        if asp > 0.60:
            explanation_lines.append(f"Surface Progression HIGH ({asp:.2f})")
        if ra > 0.60:
            explanation_lines.append(f"Route Acceleration HIGH ({ra:.2f})")
        if gnn_score > 0.60:
            explanation_lines.append(f"GNN Risk HIGH ({gnn_score:.2f})")

        exp_summary = "; ".join(explanation_lines) if explanation_lines else "Normal route progression"

        return {
            "bari_score": round(bari_score, 4),
            "route_novelty": round(rn, 4),
            "route_deviation": round(rd, 4),
            "surface_progression": round(asp, 4),
            "route_acceleration": round(ra, 4),
            "route_persistence": round(rp, 4),
            "gnn_risk": round(gnn_score, 4),
            "service_sensitivity": round(curr_sens, 4),
            "observed_route": observed_route,
            "route_string": route_str,
            "explanation_summary": exp_summary,
            "sub_metric_ratings": {
                "route_novelty": "HIGH" if rn > 0.65 else ("MEDIUM" if rn > 0.35 else "LOW"),
                "route_deviation": "HIGH" if rd > 0.65 else ("MEDIUM" if rd > 0.35 else "LOW"),
                "surface_progression": "HIGH" if asp > 0.65 else ("MEDIUM" if asp > 0.35 else "LOW"),
                "route_acceleration": "HIGH" if ra > 0.65 else ("MEDIUM" if ra > 0.35 else "LOW"),
                "route_persistence": "HIGH" if rp > 0.65 else ("MEDIUM" if rp > 0.35 else "LOW"),
                "gnn_risk": "HIGH" if gnn_score > 0.65 else ("MEDIUM" if gnn_score > 0.35 else "LOW"),
            }
        }

    def _compute_route_novelty(self, prev_key: str, curr_key: str) -> float:
        """Transition novelty vs Benign Route Memory."""
        if not self._baseline_learned or self._benign_total_flows == 0:
            cnt = self._benign_service_counts.get(curr_key, 0)
            return max(0.0, 1.0 - (cnt / max(1, self._benign_total_flows)))

        if prev_key is None:
            cnt = self._benign_service_counts.get(curr_key, 0)
            return max(0.0, 1.0 - (cnt / max(1, self._benign_total_flows)))

        trans_cnt = self._benign_transition_counts.get(prev_key, {}).get(curr_key, 0)
        prev_tot = sum(self._benign_transition_counts.get(prev_key, {}).values())
        if prev_tot == 0:
            return 0.95

        p_trans = trans_cnt / prev_tot
        return round(max(0.0, 1.0 - p_trans), 4)

    def _compute_route_deviation(self, history: deque, curr_key: str) -> float:
        """Deviation from attacker host's previously visited services."""
        if not history:
            return 0.20
        visited = [h["service_key"] for h in history]
        if curr_key in visited:
            return 0.15
        return min(1.0, 0.40 + 0.15 * len(set(visited)))

    def _compute_surface_progression(self, history: deque, curr_sens: float) -> float:
        """Progression toward higher sensitivity target services."""
        if not history:
            return curr_sens
        prev_sens = [h["sensitivity"] for h in history]
        max_prev = max(prev_sens)
        if curr_sens >= max_prev:
            return min(1.0, curr_sens + 0.10 * (curr_sens - max_prev))
        return max(curr_sens, max_prev * 0.85)

    def _compute_route_acceleration(self, history: deque, now: float) -> float:
        """Inter-arrival transition velocity."""
        if not history:
            return 0.10
        dt = max(0.001, now - history[-1]["time"])
        if dt < 0.5:
            return 1.0
        elif dt < 2.0:
            return 0.75
        elif dt < 10.0:
            return 0.40
        return 0.15

    def _compute_route_persistence(self, history: deque, curr_sens: float, rn: float) -> float:
        """Accumulated non-benign transition count across temporal window."""
        if not history:
            return 0.10
        suspicious_steps = sum(1 for h in history if h["sensitivity"] > 0.50 or rn > 0.50)
        return min(1.0, suspicious_steps / max(1, len(history)))

