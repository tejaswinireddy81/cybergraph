"""
report_generator.py
===================

Professional Incident Report Generator

Creates PDF reports for detected cyber attacks.

Contents:
- Incident ID
- Timestamp
- Source IP
- Destination IP
- Risk Score
- Risk Level
- Threat Intelligence
- Attack Chain
- AI Explanation
- Recommended Response
"""

import os
import uuid
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet


class ReportGenerator:

    def __init__(self):

        self.output_folder = "reports/generated"

        os.makedirs(
            self.output_folder,
            exist_ok=True
        )

        self.styles = getSampleStyleSheet()

    def generate(
        self,
        incident
    ):

        incident_id = str(uuid.uuid4())[:8]

        filename = os.path.join(
            self.output_folder,
            f"Incident_{incident_id}.pdf"
        )

        doc = SimpleDocTemplate(filename)

        story = []

        story.append(
            Paragraph(
                "<b>AI Network Intrusion Detection Report</b>",
                self.styles["Title"]
            )
        )

        story.append(Spacer(1, 20))

        fields = [

            ("Incident ID", incident_id),

            ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),

            ("Source IP", incident.get("src_ip", "-")),

            ("Destination IP", incident.get("dst_ip", "-")),

            ("Risk Score", incident.get("risk_score", "-")),

            ("Risk Level", incident.get("risk_level", "-")),

            ("Threat", incident.get("threat", "-")),

            ("Attack Chain",
             " -> ".join(
                 incident.get(
                     "attack_chain",
                     []
                 )
             )
            ),

            ("Recommended Action",
             incident.get(
                 "action",
                 "-"
             )
            )
        ]

        for title, value in fields:

            story.append(

                Paragraph(

                    f"<b>{title}</b>: {value}",

                    self.styles["BodyText"]

                )

            )

            story.append(Spacer(1, 8))

        reasons = incident.get(
            "explanation",
            []
        )

        story.append(

            Paragraph(

                "<b>AI Explanation</b>",

                self.styles["Heading2"]

            )

        )

        for reason in reasons:

            story.append(

                Paragraph(

                    f"• {reason}",

                    self.styles["BodyText"]

                )

            )

        doc.build(story)

        return filename