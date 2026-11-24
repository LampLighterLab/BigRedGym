"""Build the final one-page Assignment 6 project outline."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "assignment_6_project_outline.pdf"


def paragraph(text, style):
    return Paragraph(text, style)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.8,
        leading=12.8,
        spaceAfter=4,
        textColor=colors.HexColor("#161616"),
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=12.0,
        leading=13.2,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=15.5,
        leading=17.0,
        alignment=TA_CENTER,
        spaceAfter=3,
    )
    author = ParagraphStyle(
        "Author",
        parent=body,
        fontSize=9.8,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.5,
        leading=10.0,
        textColor=colors.HexColor("#333333"),
        spaceBefore=3,
    )

    story = [
        paragraph("Training-Free Composition of Go2 Walking and Height Control", title),
        paragraph("Huaijin Hu (hh833), Zoey Mo (zm73)", author),
        paragraph("Context", heading),
        paragraph(
            "<i>Expert Composer Policy</i> trains a reusable transition policy "
            "between frozen quadruped skills and reports 99.99% mean transition "
            "success in simulation and 97.22% on hardware. We ask a narrower "
            "question: when walking and height-control policies share the same 12 "
            "joint-position-target action space, can direct switching or live "
            "output blending provide useful walk-to-crouch transitions without "
            "training another policy?",
            body,
        ),
        paragraph("Contribution", heading),
        paragraph(
            "We will characterize the speed and height-change region in which "
            "training-free composition works, and identify why it fails outside "
            "that region. Policy A is frozen Go2Trot; policy B is frozen Go2Height. "
            "Each receives its own observations, then their live joint targets are "
            "switched or blended. No new policy is trained; a learned policy C is "
            "future work only.",
            body,
        ),
    ]

    evidence = paragraph(
        "<b>Preliminary evidence.</b> A survived all 30 reduced walking-gate cases "
        "but passed 0/30 because of joint jitter and gait-quality failures. At 2 "
        "Hz, velocity RMSE was 0.12 m/s for a 0.50 m/s command and 0.30 m/s for "
        "1.0 m/s. Before transitions, A reached 0.17 and 0.32 m/s from 0.20 and "
        "0.40 m/s commands. Across four gait phases per speed, B alone and direct "
        "switching each succeeded in 4/8 trials; 0.25, 0.50, and 1.00 s blends "
        "succeeded in 7/8, 6/8, and 3/8. All methods had 0.01-0.02 m steady "
        "height RMSE and at most 12.2 degrees tilt. The best 0.25 s blend still "
        "produced 0.54-0.59 rad peak joint-target steps, so it is not yet "
        "hardware-ready.",
        body,
    )
    evidence_box = Table([[evidence]], colWidths=[6.8 * inch])
    evidence_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F5F7")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#718096")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend(
        [
            evidence_box,
            paragraph("Experiment and quantitative assessment", heading),
            paragraph(
                "A and B retain separate observations while sharing joint order, "
                "action scaling, limits, and control rate. Compare B alone, "
                "immediate A-to-B switching, and "
                "0.25, 0.50, and 1.00 s linear blends of the policies' live outputs. "
                "From 0, 0.10, and 0.20 m/s, command a stop and a 0.35-to-0.20 m "
                "height change. Run at least 50 randomized-gait-phase trials per "
                "condition, then repeat the selected method with randomized mass "
                "and PD gains.",
                body,
            ),
            paragraph(
                "<b>Simulation criterion.</b> Success requires no fall and "
                "simultaneous height error <=0.025 m, forward speed <=0.10 m/s, "
                "and yaw-rate error <=0.15 rad/s within 2.0 s, sustained for 0.5 s. "
                "Report success with a 95% confidence interval, settling time, "
                "maximum roll/pitch, peak step-to-step joint-target change, and "
                "peak torque fraction. B-alone comparisons separate transition "
                "quality from B's locomotion limitation.",
                body,
            ),
            paragraph(
                "<b>Sim-to-real criterion.</b> Require >=95% success in 50 held-out "
                "randomized trials, zero falls, maximum tilt <=15 degrees, and no "
                "joint-limit request. The best current method is only 7/8 in a "
                "small deterministic test, and A fails its strict gait-quality "
                "gate, so the system is not hardware-ready. "
                "After passing, request five supervised hardware trials at <=0.10 "
                "m/s and <=0.05 m height change; require height error <=0.03 m and "
                "speed <=0.10 m/s within 3.0 s, and report peak tilt and every "
                "safety intervention.",
                body,
            ),
            KeepTogether(
                [
                    Spacer(1, 1),
                    paragraph(
                        "Reference: G. Christmann, Y.-S. Luo, and W.-C. Chen, "
                        "“Expert Composer Policy: Scalable Skill Repertoire for "
                        "Quadruped Robots,” ICRA, 2024. arXiv:2403.11412.",
                        small,
                    ),
                ]
            ),
        ]
    )

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title="Training-Free Composition of Go2 Walking and Height Control",
        author="Huaijin Hu and Zoey Mo",
        subject="MAE 5110 Assignment 6 project outline",
    )
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
