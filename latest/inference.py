"""
inference.py
============
AI-Powered Network Traffic Analyzer — INFERENCE ENGINE

Loads saved artifacts from models/ and exposes run_inference().

Fixes applied to the original scaffold:
  - corrected imports (modules.gnn -> modules.gnn_engine, modules.gat -> modules.gat)
  - self.graph.prepare_graph() previously didn't exist; now implemented for real
    in modules/graph_builder.py with genuine per-host structural features
  - GAT is scored periodically (every GAT_RESCORE_EVERY flows), not on every
    single flow -- rescoring the whole graph per-flow would be wasteful and,
    on a fast live link, would fall behind
  - score_combined now comes from the LEARNED fusion model (train_fusion.py)
    instead of a hardcoded 0.5/0.5 split
  - wires in response_engine (alerting) and adaptive_learning (threshold
    recalibration), which were previously empty stub modules

run_inference(feature_dict) -> result_dict
"""

import os
import json
import logging
import datetime
import threading
import numpy as np

from modules.behavior_analysis import BehaviorAnalyzer
from modules.graph_builder import GraphBuilder
from modules.gnn_engine import GNNEngine
from modules.fusion_engine import FusionEngine
from modules.risk_engine import RiskEngine
from modules.adaptive_learning import AdaptiveThresholdManager
from modules.threat_intelligence import ThreatIntelligence
from modules.attack_chain import AttackChainDetector
from modules.device_profile import DeviceProfile
from modules.topology_engine import TopologyEngine
from modules.explainability import ExplainabilityEngine

# This is the higher-level severity -> action policy engine (blacklist/block
# via FirewallController, incident creation via IncidentDatabase). Aliased
# on import because modules/response_engine.ResponseEngine (per-flow alert
# logging + cooldown, imported above as ResponseEngine) is a different,
# already-wired module -- see the docstring in response/response_engine.py
# for why both exist side by side.
from response.response_engine import ResponsePolicyEngine

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_RF_PATH = os.path.join(_MODELS_DIR, "rf_model.pkl")
_AE_PATH = os.path.join(_MODELS_DIR, "ae_model.keras")
_SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
_ART_PATH = os.path.join(_MODELS_DIR, "artifacts.json")

GAT_RESCORE_EVERY = 25       # rescore the host graph every N flows, not every single one
GRAPH_WINDOW_SECONDS = 300   # drop graph edges/nodes idle longer than this


class InferenceEngine:
    """Load once at startup (engine.load()), reuse for every flow."""

    def __init__(self):
        self._rf = None
        self._ae = None
        self._scaler = None
        self._feature_cols = None
        self._ae_threshold = None
        self._ae_mse_max = None
        self._attack_idx = None
        self._loaded = False

        self.behavior = BehaviorAnalyzer()
        self.graph = GraphBuilder()
        self.gnn_engine = GNNEngine()
        self.fusion = FusionEngine()
        self.risk_engine = RiskEngine()
        self.threat_intel = ThreatIntelligence()
        self.attack_chain = AttackChainDetector()
        self.device_profile = DeviceProfile()
        self.topology = TopologyEngine()
        self.adaptive = None  # created in load(), needs ae_threshold first

        # SHAP explainability -- loaded against the RF model in load(). Not
        # run on every single flow (TreeExplainer.shap_values() isn't free);
        # only for flows the heuristic risk engine calls above "Low", so the
        # live pipeline stays responsive while still explaining the flows an
        # analyst would actually want explained.
        self.explainability = ExplainabilityEngine()

        # Severity -> action policy layer: blacklist/block via
        # FirewallController (safe by default: enabled=False, dry_run=True),
        # incident creation via IncidentDatabase. Previously built but never
        # instantiated anywhere -- app.py had no Response Center and no
        # Incident History because nothing called this.
        self.policy = ResponsePolicyEngine()

        self._flow_count = 0
        self._last_gat_scores = {}

        # Guards every mutation of shared state below (self.behavior, self.graph,
        # self.attack_chain, self.device_profile, self.adaptive, and the RF/AE/GAT
        # model objects themselves). run_inference() is called concurrently from
        # two different daemon threads in live_capture.py (the packet-capture
        # thread, on TCP FIN/RST, and the flush thread, on flow timeout) -- with
        # no lock, that's a real race: dict/set/graph mutation from two threads
        # at once, and Keras/sklearn/PyTorch model objects that aren't guaranteed
        # safe for concurrent predict() calls on the same instance. This lock
        # serializes those calls so both threads can keep running independently
        # everywhere else (packet capture, flow assembly) without contending.
        self._lock = threading.RLock()

    # ── public ──────────────────────────────────────────────────────────────

    def load(self):
        import joblib
        import tensorflow as tf

        for path in (_RF_PATH, _AE_PATH, _SCALER_PATH, _ART_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Artifact not found: {path}\nRun  python train_models.py  first.")

        logger.info("Loading RF …")
        self._rf = joblib.load(_RF_PATH)

        logger.info("Loading AE …")
        self._ae = tf.keras.models.load_model(_AE_PATH)

        logger.info("Loading Scaler …")
        self._scaler = joblib.load(_SCALER_PATH)

        logger.info("Loading SHAP explainer against the RF model …")
        self.explainability.load(self._rf)

        with open(_ART_PATH) as f:
            art = json.load(f)
        self._feature_cols = art["feature_cols"]
        self._ae_threshold = float(art["ae_threshold"])
        self._ae_mse_max = float(art["ae_mse_max_train"])
        self._attack_idx = int(np.where(self._rf.classes_ == 1)[0][0])

        self.adaptive = AdaptiveThresholdManager(initial_threshold=self._ae_threshold)

        logger.info("Loading GAT (host graph) engine …")
        self.gnn_engine.load()

        self._loaded = True
        logger.info(
            "Inference engine ready. AE threshold=%.6f  AE mse_max=%.6f  GAT loaded=%s",
            self._ae_threshold, self._ae_mse_max, self.gnn_engine.loaded)

    def validate_features(self, feature_dict: dict) -> bool:
        if not self._loaded:
            return False
        missing = [c for c in self._feature_cols if c not in feature_dict]
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False
        return True

    def run_inference(self, feature_dict: dict) -> dict:
        """Thread-safe entry point -- see the _lock comment in __init__.
        Called concurrently from the capture thread and the flush thread
        in live_capture.py, so the whole scoring + shared-state-mutation
        pipeline below is serialized here."""
        with self._lock:
            return self._run_inference_locked(feature_dict)

    def _run_inference_locked(self, feature_dict: dict) -> dict:
        if not self._loaded:
            raise RuntimeError("InferenceEngine.load() has not been called.")
        if not self.validate_features(feature_dict):
            raise ValueError("Feature dict is missing required columns.")

        src_ip, dst_ip = feature_dict["_src_ip"], feature_dict["_dst_ip"]

        # ── RF (unscaled input — matches training) ────────────────────────
        import pandas as pd
        x_raw_df = pd.DataFrame([[feature_dict[c] for c in self._feature_cols]],
                                 columns=self._feature_cols)
        x_raw_df = x_raw_df.apply(lambda col: col.where(np.isfinite(col), 0.0))
        rf_prob = float(self._rf.predict_proba(x_raw_df)[0, self._attack_idx])
        rf_pred = int(self._rf.predict(x_raw_df)[0])

        # ── Autoencoder (scaled input — matches training) ─────────────────
        x_scaled = self._scaler.transform(x_raw_df).astype(np.float32)
        recon = self._ae.predict(x_scaled, verbose=0)
        ae_mse = float(np.mean(np.square(recon - x_scaled)))
        current_threshold = self.adaptive.get_threshold()
        ae_pred = int(ae_mse > current_threshold)
        ae_normalized = min(ae_mse / (self._ae_mse_max + 1e-9), 1.0)

        # feed the adaptive threshold manager -- only flows RF/GAT didn't flag
        # count toward the "probably normal" recalibration baseline
        self.adaptive.observe(ae_mse, was_flagged_by_other_detectors=bool(rf_pred))

        # ── behavior / graph bookkeeping ───────────────────────────────────
        self.behavior.update(
            src_ip,
            dst_ip,
            feature_dict["_dst_port"],
            feature_dict["_byte_count"],
            pkt_count=feature_dict.get("_pkt_count", 1),
            protocol=feature_dict.get("_protocol"),
        )
        behavior_profile = self.behavior.get_profile(src_ip)

        self.graph.add_connection(
            src_ip, dst_ip,
            dst_port=feature_dict.get("_dst_port", 0),
            protocol=feature_dict.get("_protocol", 0),
            total_bytes=feature_dict.get("_byte_count", 0),
            total_packets=feature_dict.get("_pkt_count", 0),
            retransmit=int(feature_dict.get("RETRANSMITTED_IN_BYTES", 0)
                            + feature_dict.get("RETRANSMITTED_OUT_BYTES", 0)),
        )
        self.topology.update(src_ip, dst_ip)
        network_graph = self.graph.get_graph()

        # ── GAT: periodic re-score, not per-flow (efficiency) ─────────────
        self._flow_count += 1
        if self._flow_count % GAT_RESCORE_EVERY == 0:
            self.graph.prune_older_than(GRAPH_WINDOW_SECONDS)
            try:
                self._last_gat_scores = self.gnn_engine.score_graph(self.graph)
            except Exception as e:
                logger.warning(f"GAT scoring error: {e}")
        gat_scored = src_ip in self._last_gat_scores
        gat_score = self._last_gat_scores.get(src_ip, 0.0)

        # ── learned fusion (falls back to fixed weights only if untrained) ─
        # BUGFIX: the learned 3-input model was trained expecting gat_score to
        # be a real signal. Before this host's first GAT rescore, gat_score is
        # a placeholder 0.0 -- but the model reads "gat=0.0" as confident
        # evidence of *benign*, not "unknown". With this model's fitted
        # weights (rf≈7.98, ae≈0.17, gat≈8.06, intercept≈-8.04), that means
        # even a 100%-confidence RF attack call, fused with an unscored
        # gat=0.0, tops out around 0.485 -- never reaching "High Risk", no
        # matter how certain RF is. Until GAT has actually scored this host
        # at least once, fall back to an RF/AE-only heuristic instead of
        # feeding the learned model a fabricated "benign" GAT signal.
        if gat_scored or self.fusion.using_fallback:
            fusion_result = self.fusion.predict(rf_prob, ae_normalized, gat_score)
        else:
            interim_score = min(1.0, 0.75 * rf_prob + 0.25 * ae_normalized)
            fusion_result = {
                "score": round(interim_score, 4),
                "risk": ("High Risk" if interim_score >= 0.70 else
                          "Medium Risk" if interim_score >= 0.40 else "Low Risk"),
                "using_fallback_weights": True,
            }
        score_combined = fusion_result["score"]

        # ── explainable 0-100 heuristic score (separate from the ML fusion
        #    score -- this one exists for human SOC-analyst readability) ───
        risk_0_100, risk_level = self.risk_engine.calculate(
            rf_prob, ae_normalized,
            behavior_profile["packets"],
            len(behavior_profile["destinations"]),
            len(behavior_profile["ports"]),
        )

        # ── threat intel / attack-chain / device profile ──────────────────
        intel = self.threat_intel.lookup(src_ip)
        predicted_attack_name = "Attack" if rf_pred else "Normal"
        self.attack_chain.update(src_ip, predicted_attack_name)
        chain = self.attack_chain.detect(src_ip)
        ts = datetime.datetime.fromtimestamp(feature_dict["_timestamp"])

        # ── explainability: human-readable reasons always; real SHAP
        #    per-feature contributions only for flows the heuristic risk
        #    engine calls above "Low" (TreeExplainer.shap_values() isn't
        #    free -- gating it keeps the live pipeline responsive) ────────
        explanation_summary = self.explainability.generate_summary(
            rf_prob, ae_normalized, gat_score, score_combined)
        shap_top_features = None
        if risk_level != "Low":
            try:
                shap_top_features = self.explainability.explain_rf(
                    self._rf, self._feature_cols, x_raw_df)
            except Exception as e:
                logger.warning(f"SHAP explanation error: {e}")
        explanation = {"summary": explanation_summary, "shap_top_features": shap_top_features}

        # ── severity -> action policy (blacklist/block/incident) ──────────
        # risk_level here ("Low"/"Medium"/"High"/"Critical") comes from the
        # heuristic RiskEngine above -- it's the one score in this pipeline
        # that already speaks the same 4-level vocabulary the policy engine
        # expects, so it's what decides severity here. This is a genuinely
        # live, automatic path: every flow above Low creates/updates an
        # incident and (per the severity table in response/response_engine.py)
        # may blacklist or block src_ip -- but real firewall enforcement stays
        # off unless FirewallController.enabled/dry_run are explicitly flipped.
        policy_result = self.policy.handle(
            src_ip=src_ip,
            severity=risk_level,
            dst_ip=dst_ip,
            risk_score=score_combined,
            confidence=rf_prob,
            src_port=feature_dict["_src_port"],
            dst_port=feature_dict["_dst_port"],
            protocol=feature_dict.get("_protocol"),
            pkt_count=feature_dict.get("_pkt_count"),
            byte_count=feature_dict.get("_byte_count"),
            duration=feature_dict.get("_duration"),
            attack_type=predicted_attack_name,
            rf_prediction=rf_pred,
            rf_confidence=rf_prob,
            ae_score=ae_normalized,
            gnn_score=None,
            gat_score=gat_score,
            explanation=explanation,
            attack_chain=chain,
            threat_intel_status=intel["status"],
            event_timestamp=ts.isoformat(),
        )

        top_peers = [{"ip": ip, "bytes": b} for ip, b in self.behavior.top_peers(src_ip, n=5)]
        protocols = self.behavior.protocol_distribution(src_ip)

        is_blocked_or_blacklisted = (
            self.policy.firewall.is_blocked(src_ip)
            or self.policy.firewall.is_blacklisted(src_ip)
        )
        self.device_profile.update(
            src_ip,
            packets=feature_dict.get("_pkt_count", 0),
            bytes_count=feature_dict.get("_byte_count", 0),
            rf_score=rf_prob,
            ae_score=ae_normalized,
            gat_score=gat_score,
            fusion_score=score_combined,
            risk_score=risk_0_100,
            risk_level=risk_level,
            threat=predicted_attack_name,
            attack_chain=chain,
            status="Blocked" if is_blocked_or_blacklisted else "Active",
            is_local=behavior_profile.get("is_local"),
            protocols=protocols,
            top_peers=top_peers,
            in_packets=behavior_profile.get("in_packets", 0),
            in_bytes=behavior_profile.get("in_bytes", 0),
            out_packets=behavior_profile.get("out_packets", 0),
            out_bytes=behavior_profile.get("out_bytes", 0),
            explanation=explanation,
            response_action=policy_result.get("action"),
            incident_id=policy_result.get("incident_id"),
        )

        return {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip": src_ip, "dst_ip": dst_ip,
            "src_port": feature_dict["_src_port"], "dst_port": feature_dict["_dst_port"],
            "protocol": feature_dict["_protocol"],
            "pkt_count": feature_dict["_pkt_count"], "byte_count": feature_dict["_byte_count"],
            "duration": round(feature_dict["_duration"], 4),
            "pkt_rate": round(feature_dict["_pkt_rate"], 4),
            "device_packets": behavior_profile["packets"],
            "device_bytes": behavior_profile["bytes"],
            "device_flows": behavior_profile["flows"],
            "unique_destinations": len(behavior_profile["destinations"]),
            "unique_ports": len(behavior_profile["ports"]),
            "graph_nodes": network_graph.number_of_nodes(),
            "graph_edges": network_graph.number_of_edges(),
            "gat_score": round(gat_score, 4),
            "threat_intel_status": intel["status"],
            "attack_chain": chain,
            "true_label": "N/A — Live Traffic",
            "true_label_enc": -1,
            "rf_pred": rf_pred, "rf_prob": round(rf_prob, 6),
            "ae_mse": round(ae_mse, 6), "ae_pred": ae_pred,
            "ae_threshold_current": round(current_threshold, 6),
            "score_combined": round(score_combined, 6),
            "risk_level_ml": fusion_result["risk"],
            "risk_score_heuristic": risk_0_100,
            "risk_level_heuristic": risk_level,
            "response_action": policy_result.get("action"),
            "using_fallback_fusion_weights": fusion_result["using_fallback_weights"],
            # ── Priority 4/5/6/7 additions ──────────────────────────────
            "is_local_src": self.behavior.get_profile(src_ip).get("is_local"),
            "is_local_dst": self.behavior.devices[dst_ip]["is_local"] if dst_ip in self.behavior.devices else None,
            "explanation_summary": explanation_summary,
            "shap_top_features": shap_top_features,
            "policy_action": policy_result.get("action"),
            "policy_incident_id": policy_result.get("incident_id"),
            "policy_suppressed": policy_result.get("suppressed", False),
            "firewall_enabled": self.policy.firewall.enabled,
            "firewall_dry_run": self.policy.firewall.dry_run,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
engine = InferenceEngine()
