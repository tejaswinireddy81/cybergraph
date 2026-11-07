"""
incident_report.py
===================
Section 7 of the requirements doc: CSV export for structured analysis,
and a PDF forensic report for a single incident or a date range.

Reads from database/incident_db.py — this module has no knowledge of
live inference at all, it only formats what's already been recorded.
That separation matters: forensic reports should reflect exactly what
was stored at detection time, not be re-derived from current model
state.

PDF layout, per incident:
  - Attack summary   (who/what/when/severity)
  - Traffic evidence  (IPs, ports, protocol, packet/byte stats, duration)
  - AI outputs        (RF, Autoencoder, GNN, GAT, fused risk score)
  - Explanation        (SHAP/LIME contributions + human-readable reasons)
  - Timeline            (created, event time, resolution time)
  - Response action      (what was done, current status)
"""

import os
import csv
import json
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

from database.incident_db import IncidentDatabase

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")

_CSV_FIELDS = [
    "incident_id", "created_at", "event_timestamp", "severity",
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "pkt_count", "byte_count", "duration",
    "attack_type", "attack_chain",
    "rf_prediction", "rf_confidence", "ae_score", "gnn_score", "gat_score",
    "final_risk_score", "risk_score_scale",
    "threat_intel_status",
    "response_action", "response_status", "resolution_time",
]


def _fmt(value, default="-"):
    if value is None or value == "":
        return default
    return value


class IncidentReporter:
    def __init__(self, incident_db: IncidentDatabase = None, output_dir: str = _OUTPUT_DIR):
        self.db = incident_db or IncidentDatabase()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

    # ── CSV export ───────────────────────────────────────────────────────

    def generate_csv(
        self,
        incident_ids: list = None,
        severity=None,
        start_date=None,
        end_date=None,
        output_path: str = None,
    ) -> str:
        """Exports incidents (filtered, or by explicit ID list) to CSV.
        Returns the path written."""
        if incident_ids:
            rows = [self.db.get_incident(i) for i in incident_ids]
            rows = [r for r in rows if r is not None]
        else:
            rows = self.db.query_incidents(
                severity=severity, start_date=start_date, end_date=end_date, limit=100000
            )

        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"incidents_{stamp}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return output_path

    # ── PDF: single incident forensic report ────────────────────────────

    def generate_pdf(self, incident_id: str, output_path: str = None) -> str:
        incident = self.db.get_incident(incident_id)
        if incident is None:
            raise ValueError(f"No incident found with id {incident_id!r}")

        if output_path is None:
            output_path = os.path.join(self.output_dir, f"Incident_{incident_id}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        story.append(Paragraph("AI Network Intrusion Detection — Forensic Report", self.styles["Title"]))
        story.append(Spacer(1, 16))
        self._append_incident_sections(story, incident)
        doc.build(story)
        return output_path

    # ── PDF: date-range forensic report (multiple incidents) ───────────

    def generate_pdf_range(
        self, start_date, end_date, severity=None, output_path: str = None
    ) -> str:
        incidents = self.db.query_incidents(
            severity=severity, start_date=start_date, end_date=end_date, limit=100000, order="ASC"
        )

        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"Incidents_{stamp}.pdf")

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        story.append(Paragraph("AI Network Intrusion Detection — Forensic Report", self.styles["Title"]))
        story.append(Paragraph(f"Range: {start_date or '(earliest)'} to {end_date or '(latest)'}",
                                self.styles["Normal"]))
        story.append(Spacer(1, 12))

        counts = {}
        for inc in incidents:
            counts[inc["severity"]] = counts.get(inc["severity"], 0) + 1
        story.append(Paragraph(
            f"<b>{len(incidents)} incidents</b> — " +
            ", ".join(f"{k}: {v}" for k, v in counts.items()),
            self.styles["Normal"]
        ))
        story.append(Spacer(1, 20))

        for i, incident in enumerate(incidents):
            if i > 0:
                story.append(PageBreak())
            story.append(Paragraph(f"Incident {incident['incident_id']}", self.styles["Heading1"]))
            story.append(Spacer(1, 8))
            self._append_incident_sections(story, incident)

        doc.build(story)
        return output_path

    # ── shared section builder ─────────────────────────────────────────

    def _append_incident_sections(self, story, incident):
        table_style = TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])

        # -- Attack summary --
        story.append(Paragraph("Attack Summary", self.styles["Heading2"]))
        summary_rows = [
            ["Incident ID", _fmt(incident.get("incident_id"))],
            ["Severity", _fmt(incident.get("severity"))],
            ["Attack Type", _fmt(incident.get("attack_type"))],
            ["Attack Chain", _fmt(incident.get("attack_chain"))],
            ["Source IP", _fmt(incident.get("src_ip"))],
            ["Destination IP", _fmt(incident.get("dst_ip"))],
            ["Threat Intel Status", _fmt(incident.get("threat_intel_status"))],
        ]
        story.append(Table(summary_rows, colWidths=[1.8 * inch, 4.2 * inch], style=table_style))
        story.append(Spacer(1, 14))

        # -- Traffic evidence --
        story.append(Paragraph("Traffic Evidence", self.styles["Heading2"]))
        traffic_rows = [
            ["Source Port", _fmt(incident.get("src_port"))],
            ["Destination Port", _fmt(incident.get("dst_port"))],
            ["Protocol", _fmt(incident.get("protocol"))],
            ["Packet Count", _fmt(incident.get("pkt_count"))],
            ["Byte Count", _fmt(incident.get("byte_count"))],
            ["Duration (s)", _fmt(incident.get("duration"))],
        ]
        story.append(Table(traffic_rows, colWidths=[1.8 * inch, 4.2 * inch], style=table_style))
        story.append(Spacer(1, 14))

        # -- AI outputs --
        story.append(Paragraph("AI Model Outputs", self.styles["Heading2"]))
        ai_rows = [
            ["Random Forest Prediction", _fmt(incident.get("rf_prediction"))],
            ["Random Forest Confidence", _fmt(incident.get("rf_confidence"))],
            ["Autoencoder Score", _fmt(incident.get("ae_score"))],
            ["GNN Score", _fmt(incident.get("gnn_score"))],
            ["GAT Score", _fmt(incident.get("gat_score"))],
            ["Final Fused Risk Score", _fmt(incident.get("final_risk_score"))],
            ["Risk Score Scale", _fmt(incident.get("risk_score_scale"))],
        ]
        story.append(Table(ai_rows, colWidths=[1.8 * inch, 4.2 * inch], style=table_style))
        story.append(Spacer(1, 14))

        # -- Explanation (SHAP/LIME) --
        story.append(Paragraph("Explanation", self.styles["Heading2"]))
        explanation = incident.get("explanation")
        reasons = self._parse_explanation(explanation)
        if reasons:
            for reason in reasons:
                story.append(Paragraph(f"• {reason}", self.styles["BodyText"]))
        else:
            story.append(Paragraph("No explanation recorded for this incident.", self.styles["BodyText"]))
        story.append(Spacer(1, 14))

        # -- Timeline --
        story.append(Paragraph("Timeline", self.styles["Heading2"]))
        timeline_rows = [
            ["Event Occurred", _fmt(incident.get("event_timestamp"))],
            ["Incident Recorded", _fmt(incident.get("created_at"))],
            ["Resolved", _fmt(incident.get("resolution_time"), default="Not yet resolved")],
        ]
        story.append(Table(timeline_rows, colWidths=[1.8 * inch, 4.2 * inch], style=table_style))
        story.append(Spacer(1, 14))

        # -- Response action --
        story.append(Paragraph("Response Action", self.styles["Heading2"]))
        response_rows = [
            ["Action Taken", _fmt(incident.get("response_action"))],
            ["Status", _fmt(incident.get("response_status"))],
        ]
        story.append(Table(response_rows, colWidths=[1.8 * inch, 4.2 * inch], style=table_style))

    @staticmethod
    def _parse_explanation(explanation) -> list:
        """explanation may be: None, a JSON-encoded list of strings, a
        JSON-encoded list of {feature, impact} dicts, or a plain string."""
        if not explanation:
            return []
        if isinstance(explanation, list):
            data = explanation
        else:
            try:
                data = json.loads(explanation)
            except (json.JSONDecodeError, TypeError):
                return [str(explanation)]

        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "feature" in item:
                    out.append(f"{item['feature']}: impact {item.get('impact', 0):+.4f}")
                else:
                    out.append(str(item))
            return out
        return [str(data)]
