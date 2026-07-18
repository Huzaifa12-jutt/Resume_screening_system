"""
cv_generator.py
----------------
Generates ready-made sample resume PDFs so the app can be demoed instantly
without needing real resumes. Two profiles are created:

    1. John Anderson   -> strong AI/ML Engineer profile
    2. Sarah Williams  -> strong Full-Stack Developer profile (weaker ML fit)

These match the "Sample Test Scenario" described in the assignment brief,
so when you run them against the default Senior AI/ML Engineer job
description, John should rank #1 and Sarah #2.

The PDFs are built with reportlab and written to a temp folder, then
returned as raw bytes so Streamlit can hand them straight to the parser
(exactly like an uploaded file).
"""

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


def _build_pdf(sections: dict) -> bytes:
    """Turn a dict of {heading: text} into a simple, clean one-page PDF."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "NameStyle", parent=styles["Title"], fontSize=18,
        textColor=colors.HexColor("#1E293B"), alignment=TA_LEFT, spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "ContactStyle", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#475569"), spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontSize=12,
        textColor=colors.HexColor("#0F766E"), spaceBefore=10, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=9.5, leading=14,
    )

    story = [Paragraph(sections["name"], name_style),
             Paragraph(sections["contact"], contact_style)]

    for heading, text in sections["body"]:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(text.replace("\n", "<br/>"), body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    return buffer.getvalue()


def generate_ai_ml_cv() -> bytes:
    """AI/ML Engineer profile: John Anderson."""
    sections = {
        "name": "John Anderson",
        "contact": "john.anderson@example.com | +1-555-0142 | Karachi, Pakistan | linkedin.com/in/johnanderson",
        "body": [
            ("Summary",
             "AI/ML Engineer with 6 years of experience building and deploying machine "
             "learning and deep learning systems in production. Specialised in NLP, "
             "computer vision, and MLOps on cloud infrastructure."),
            ("Technical Skills",
             "Python, TensorFlow, PyTorch, Scikit-learn, LangChain, NLP, Computer Vision, "
             "AWS, GCP, Docker, Kubernetes, FastAPI, PostgreSQL, Git, CI/CD, Pandas, NumPy"),
            ("Professional Experience",
             "Senior Machine Learning Engineer, Vertex Analytics (2021 - Present)\n"
             "- Led a 5-year initiative building RAG-based document search systems using LangChain.\n"
             "- Deployed deep learning models (TensorFlow, PyTorch) to AWS SageMaker at scale.\n"
             "- Built MLOps pipelines with Docker and Kubernetes cutting deployment time by 40%.\n\n"
             "Machine Learning Engineer, DataCore Labs (2019 - 2021)\n"
             "- Developed NLP and computer vision models for enterprise clients.\n"
             "- Managed data pipelines with PostgreSQL and GCP."),
            ("Education",
             "MS in Computer Science (Machine Learning), FAST-NUCES, 2019\n"
             "BS in Computer Science, Air University, 2017"),
            ("Certifications",
             "AWS Certified Machine Learning - Specialty\n"
             "TensorFlow Developer Certificate\n"
             "Google Cloud Professional ML Engineer"),
            ("Projects",
             "RAG-based Enterprise Knowledge Assistant - LangChain + FastAPI + PostgreSQL, "
             "deployed on AWS with Docker/Kubernetes.\n"
             "Real-time Object Detection System - PyTorch + Computer Vision pipeline for "
             "manufacturing quality control.\n"
             "Led a small team (team leadership) delivering the above MLOps platform."),
        ],
    }
    return _build_pdf(sections)


def generate_fullstack_cv() -> bytes:
    """Full-Stack Developer profile: Sarah Williams (weaker AI/ML fit on purpose)."""
    sections = {
        "name": "Sarah Williams",
        "contact": "sarah.williams@example.com | +1-555-0198 | Lahore, Pakistan | linkedin.com/in/sarahwilliams",
        "body": [
            ("Summary",
             "Full-Stack Developer with 4 years of experience building scalable web "
             "applications. Comfortable across the stack with growing exposure to "
             "cloud deployment and basic data tooling."),
            ("Technical Skills",
             "JavaScript, TypeScript, React, Node.js, MongoDB, SQL, AWS, Docker, Git, "
             "Python, Flask, Redis"),
            ("Professional Experience",
             "Full-Stack Developer, BrightWorks Software (2021 - Present)\n"
             "- Built and maintained React/Node.js web applications for retail clients.\n"
             "- Integrated REST APIs and MongoDB/SQL databases.\n"
             "- Deployed services using Docker on AWS EC2.\n\n"
             "Junior Web Developer, PixelForge Studio (2020 - 2021)\n"
             "- Developed responsive front-end interfaces using React."),
            ("Education",
             "BS in Software Engineering, University of the Punjab, 2020"),
            ("Certifications",
             "AWS Certified Cloud Practitioner"),
            ("Projects",
             "E-commerce Platform - React, Node.js, MongoDB, deployed with Docker on AWS.\n"
             "Internal Analytics Dashboard - Python/Flask backend with basic data "
             "visualisation for sales reporting."),
        ],
    }
    return _build_pdf(sections)


SAMPLE_CV_GENERATORS = {
    "John Anderson - AI/ML Engineer CV": generate_ai_ml_cv,
    "Sarah Williams - Full Stack Developer CV": generate_fullstack_cv,
}


if __name__ == "__main__":
    # Quick manual test: writes both sample PDFs to ./sample_data/
    import os
    os.makedirs("sample_data", exist_ok=True)
    for label, fn in SAMPLE_CV_GENERATORS.items():
        path = os.path.join("sample_data", label.split(" - ")[0].replace(" ", "_") + ".pdf")
        with open(path, "wb") as f:
            f.write(fn())
        print("Wrote", path)
