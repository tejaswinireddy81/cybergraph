"""
incident_db.py
===============
Persistent incident store for the NIDS.

Every significant detection (Medium severity and above — see
response/response_engine.py) gets a row here. This is the system of
record the reporting layer (reporting/incident_report.py) reads from,
and what the dashboard's "Incident History" panel queries.

Schema follows Section 6 of the requirements doc: incident ID,
timestamp, source/destination IP + ports + protocol, packet/byte
stats, attack type, RF prediction + confidence, Autoencoder score,
GNN/GAT outputs (when available), final risk score, severity,
SHAP/LIME explanation, response action, response status, resolution
time.

Backend: SQLite. That's a deliberate choice, not a placeholder —
a single-host NIDS writing incident-rate volumes of rows (not
per-packet, not per-flow) has no need for a client/server database,
and SQLite gives durability + SQL filtering with zero extra
infrastructure. WAL mode is enabled so the dashboard can read while
inference.py is writing.
"""

import os
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime

_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DEFAULT_DB_PATH = os.path.join(_DB_DIR, "incidents.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id         TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,        -- ISO8601, when the row was written
    event_timestamp     TEXT,                 -- ISO8601, when the flow/event occurred

    src_ip              TEXT NOT NULL,
    dst_ip              TEXT,
    src_port            INTEGER,
    dst_port            INTEGER,
    protocol            TEXT,

    pkt_count           INTEGER,
    byte_count          INTEGER,
    duration            REAL,

    attack_type         TEXT,                 -- e.g. "Port Scan", "Brute Force", "Normal"

    rf_prediction        INTEGER,             -- 0/1
    rf_confidence         REAL,               -- 0..1
    ae_score              REAL,               -- 0..1 (normalized reconstruction error)
    gnn_score              REAL,              -- 0..1, NULL if GNN/GAT not available for this event
    gat_score               REAL,             -- 0..1, NULL if GAT not available for this event

    final_risk_score        REAL,             -- fused 0..1 (or 0..100 heuristic, see risk_score_scale)
    risk_score_scale          TEXT DEFAULT '0-1',
    severity                    TEXT NOT NULL,   -- Low | Medium | High | Critical

    explanation                  TEXT,         -- JSON: SHAP/LIME contributions + human-readable reasons
    attack_chain                    TEXT,      -- e.g. "Port Scan -> Brute Force"
    threat_intel_status               TEXT,

    response_action                     TEXT,  -- e.g. "logged", "alerted", "blacklisted", "blocked"
    response_status                       TEXT DEFAULT 'open',  -- open | acknowledged | resolved | false_positive
    resolution_time                         TEXT,  -- ISO8601, set when resolved

    raw_details                               TEXT   -- JSON: anything else worth keeping (full inference result)
);

CREATE INDEX IF NOT EXISTS idx_incidents_created_at ON incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_incidents_src_ip     ON incidents(src_ip);
CREATE INDEX IF NOT EXISTS idx_incidents_severity   ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_status     ON incidents(response_status);
"""

_VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}
_VALID_STATUSES = {"open", "acknowledged", "resolved", "false_positive"}


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


class IncidentDatabase:
    """Thread-safe SQLite wrapper. One instance is safe to share across
    the capture/inference threads and the Dash callback threads."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
        finally:
            conn.close()

    # ── write ──────────────────────────────────────────────────────────

    def create_incident(
        self,
        src_ip,
        severity,
        dst_ip=None,
        src_port=None,
        dst_port=None,
        protocol=None,
        pkt_count=None,
        byte_count=None,
        duration=None,
        attack_type=None,
        rf_prediction=None,
        rf_confidence=None,
        ae_score=None,
        gnn_score=None,
        gat_score=None,
        final_risk_score=None,
        risk_score_scale="0-1",
        explanation=None,
        attack_chain=None,
        threat_intel_status=None,
        response_action=None,
        response_status="open",
        event_timestamp=None,
        raw_details=None,
        incident_id=None,
    ) -> str:
        """Insert one incident row. Returns the incident_id.

        `explanation` and `raw_details` may be a dict/list (will be
        JSON-serialized) or a pre-serialized string.
        """
        if severity not in _VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {_VALID_SEVERITIES}, got {severity!r}")
        if response_status not in _VALID_STATUSES:
            raise ValueError(f"response_status must be one of {_VALID_STATUSES}, got {response_status!r}")

        incident_id = incident_id or f"INC-{uuid.uuid4().hex[:10]}"

        def _to_json(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            return json.dumps(value, default=str)

        row = {
            "incident_id": incident_id,
            "created_at": _now_iso(),
            "event_timestamp": event_timestamp,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "pkt_count": pkt_count,
            "byte_count": byte_count,
            "duration": duration,
            "attack_type": attack_type,
            "rf_prediction": rf_prediction,
            "rf_confidence": rf_confidence,
            "ae_score": ae_score,
            "gnn_score": gnn_score,
            "gat_score": gat_score,
            "final_risk_score": final_risk_score,
            "risk_score_scale": risk_score_scale,
            "severity": severity,
            "explanation": _to_json(explanation),
            "attack_chain": attack_chain,
            "threat_intel_status": threat_intel_status,
            "response_action": response_action,
            "response_status": response_status,
            "resolution_time": None,
            "raw_details": _to_json(raw_details),
        }

        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        with self._lock, self._connect() as conn:
            conn.execute(f"INSERT INTO incidents ({cols}) VALUES ({placeholders})", row)
            conn.commit()
        return incident_id

    def update_response(self, incident_id, response_action=None, response_status=None, resolved=False):
        """Update an existing incident after a response action (e.g. block
        succeeded, analyst marked it resolved/false-positive)."""
        if response_status is not None and response_status not in _VALID_STATUSES:
            raise ValueError(f"response_status must be one of {_VALID_STATUSES}, got {response_status!r}")

        fields, params = [], {"incident_id": incident_id}
        if response_action is not None:
            fields.append("response_action = :response_action")
            params["response_action"] = response_action
        if response_status is not None:
            fields.append("response_status = :response_status")
            params["response_status"] = response_status
        if resolved or response_status == "resolved":
            fields.append("resolution_time = :resolution_time")
            params["resolution_time"] = _now_iso()

        if not fields:
            return False

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE incidents SET {', '.join(fields)} WHERE incident_id = :incident_id", params
            )
            conn.commit()
            return cur.rowcount > 0

    # ── read ───────────────────────────────────────────────────────────

    def get_incident(self, incident_id) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
            ).fetchone()
            return dict(row) if row else None

    def query_incidents(
        self,
        severity=None,
        src_ip=None,
        response_status=None,
        start_date=None,   # inclusive, ISO8601 or 'YYYY-MM-DD'
        end_date=None,     # inclusive, ISO8601 or 'YYYY-MM-DD'
        limit=500,
        order="DESC",
    ) -> list[dict]:
        """Flexible filtered search, used by the Incident History panel
        and by the reporting layer for date-range exports."""
        clauses, params = [], {}
        if severity:
            severities = [severity] if isinstance(severity, str) else list(severity)
            placeholders = ", ".join(f":sev{i}" for i in range(len(severities)))
            clauses.append(f"severity IN ({placeholders})")
            params.update({f"sev{i}": s for i, s in enumerate(severities)})
        if src_ip:
            clauses.append("src_ip = :src_ip")
            params["src_ip"] = src_ip
        if response_status:
            clauses.append("response_status = :response_status")
            params["response_status"] = response_status
        if start_date:
            clauses.append("created_at >= :start_date")
            params["start_date"] = start_date
        if end_date:
            clauses.append("created_at <= :end_date")
            params["end_date"] = end_date

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if str(order).upper() != "ASC" else "ASC"
        sql = f"SELECT * FROM incidents {where} ORDER BY created_at {order} LIMIT :limit"
        params["limit"] = int(limit)

        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def count_by_severity(self) -> dict:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT severity, COUNT(*) AS n FROM incidents GROUP BY severity"
            ).fetchall()
            counts = {s: 0 for s in _VALID_SEVERITIES}
            counts.update({r["severity"]: r["n"] for r in rows})
            return counts

    def delete_incident(self, incident_id) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM incidents WHERE incident_id = ?", (incident_id,))
            conn.commit()
            return cur.rowcount > 0
