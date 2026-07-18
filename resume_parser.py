"""
resume_parser.py
-----------------
Turns a raw resume PDF into structured data:
    name, email, phone, skills, education, experience,
    certifications, projects, raw_text

Approach (kept deliberately simple + dependency-light, per the brief):
    1. PyPDF2 pulls raw text out of the PDF.
    2. Regex finds contact details (email / phone).
    3. The first non-empty line is treated as the candidate's name.
    4. A keyword list of 60+ technologies is scanned against the text to
       detect skills (case-insensitive, word-boundary aware).
    5. Section headers (EDUCATION, EXPERIENCE, CERTIFICATIONS, PROJECTS...)
       are used to slice the resume into blocks of relevant text.
"""

import re
import PyPDF2


# ---------------------------------------------------------------------------
# 60+ technology keywords grouped by category (matches the brief's list)
# ---------------------------------------------------------------------------
SKILL_CATEGORIES = {
    "Programming": ["Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
                     "Ruby", "Go", "PHP", "Kotlin", "Swift"],
    "Web": ["React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
            "FastAPI", "HTML", "CSS", "Next.js", "Express"],
    "AI/ML": ["TensorFlow", "PyTorch", "Scikit-learn", "LangChain", "NLP",
              "Computer Vision", "Keras", "OpenCV", "Hugging Face",
              "Deep Learning", "Machine Learning", "MLOps", "RAG",
              "LLM", "Transformers"],
    "Cloud": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
              "Lambda", "SageMaker", "Heroku", "DigitalOcean"],
    "Databases": ["SQL", "PostgreSQL", "MongoDB", "Redis", "MySQL",
                  "SQLite", "Firebase", "DynamoDB", "Elasticsearch"],
    "Tools": ["Git", "Jenkins", "CI/CD", "Linux", "Jira", "Postman",
              "GitHub Actions", "Bash", "Agile", "Scrum"],
    "Data": ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau",
              "Power BI", "Spark", "Airflow"],
}

# Flat list used for scanning, longest-first so "Computer Vision" matches
# before a bare "Vision" style false-positive would ever be a concern.
ALL_SKILLS = sorted(
    {s for group in SKILL_CATEGORIES.values() for s in group},
    key=len, reverse=True
)

SECTION_HEADERS = {
    "education": ["EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS"],
    "experience": ["EXPERIENCE", "WORK HISTORY", "PROFESSIONAL EXPERIENCE",
                   "EMPLOYMENT"],
    "certifications": ["CERTIFICATIONS", "CERTIFICATES", "CREDENTIALS"],
    "projects": ["PROJECTS", "NOTABLE PROJECTS", "KEY PROJECTS"],
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-.\s()]{7,}\d")


def extract_text_from_pdf(file_bytes) -> str:
    """Extract raw text from a PDF given its bytes (or a file-like object)."""
    reader = PyPDF2.PdfReader(file_bytes)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_name(text: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        # Skip empty lines and lines that look like contact info
        if clean and not EMAIL_RE.search(clean) and not PHONE_RE.search(clean):
            if 2 <= len(clean.split()) <= 5 and len(clean) < 50:
                return clean
    return "Unknown Candidate"


def _extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else "Not found"


def _extract_phone(text: str) -> str:
    match = PHONE_RE.search(text)
    return match.group(0).strip() if match else "Not found"


def _extract_skills(text: str) -> list:
    found = []
    lower_text = text.lower()
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lower_text):
            found.append(skill)
    # Preserve a stable, readable order (by category)
    ordered = [s for group in SKILL_CATEGORIES.values() for s in group if s in found]
    return ordered


def _extract_section(text: str, header_variants: list) -> str:
    """Grab the block of text following any of the given section headers,
    stopping at the next known section header."""
    lines = text.splitlines()
    all_headers = [h for headers in SECTION_HEADERS.values() for h in headers]

    start_idx = None
    for i, line in enumerate(lines):
        if any(h.lower() in line.strip().lower() and len(line.strip()) < 40 for h in header_variants):
            start_idx = i + 1
            break
    if start_idx is None:
        return ""

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if any(h.lower() == lines[j].strip().lower() for h in all_headers):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def parse_resume(file_bytes, filename: str = "resume.pdf") -> dict:
    """Main entry point: PDF bytes -> structured candidate dict."""
    raw_text = extract_text_from_pdf(file_bytes)

    return {
        "filename": filename,
        "name": _extract_name(raw_text),
        "email": _extract_email(raw_text),
        "phone": _extract_phone(raw_text),
        "skills": _extract_skills(raw_text),
        "education": _extract_section(raw_text, SECTION_HEADERS["education"]),
        "experience": _extract_section(raw_text, SECTION_HEADERS["experience"]),
        "certifications": _extract_section(raw_text, SECTION_HEADERS["certifications"]),
        "projects": _extract_section(raw_text, SECTION_HEADERS["projects"]),
        "raw_text": raw_text,
    }
