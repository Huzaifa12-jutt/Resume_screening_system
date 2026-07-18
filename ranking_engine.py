"""
ranking_engine.py
------------------
Combines skill overlap + TF-IDF/cosine semantic similarity into the
five-category weighted score described in the assignment brief:

    Skill Match          35%
    Experience Match      30%
    Education Match       15%
    Certification Match   10%
    Project Relevance     10%

Each sub-score is 0-100. The overall score is the weighted sum, so it is
naturally 0-100 as well.
"""

import re
from resume_parser import ALL_SKILLS
from vector_store import ResumeVectorStore

WEIGHTS = {
    "skills": 0.35,
    "experience": 0.30,
    "education": 0.15,
    "certifications": 0.10,
    "projects": 0.10,
}

DEGREE_KEYWORDS = ["phd", "doctorate", "master", "ms ", "m.s", "msc",
                    "bachelor", "bs ", "b.s", "bsc"]

RELEVANT_FIELDS = ["computer science", "software", "engineering",
                    "information technology", "data science",
                    "artificial intelligence", "mathematics", "statistics"]

# Raw TF-IDF cosine similarity on short resume sections rarely climbs above
# ~0.25-0.30 even for a genuinely strong match (short documents share few
# exact tokens). This scale factor was chosen empirically during the EDA
# notebook (see notebook.ipynb, "Calibrating the similarity scale") so
# that a clearly-relevant resume lands in the 80-100 range instead of being
# compressed near the bottom of the scale.
SEMANTIC_SCALE_FACTOR = 3.0


def _extract_required_skills(job_description: str) -> list:
    """Same 60+ keyword scan used on resumes, applied to the JD so we know
    which skills actually matter for this role."""
    lower = job_description.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lower):
            found.append(skill)
    return found


def _skill_score(candidate_skills: list, required_skills: list) -> tuple:
    if not required_skills:
        # If the JD lists no recognisable skills, fall back to "has any skills at all"
        return (min(len(candidate_skills) * 10, 100), [], candidate_skills)
    matched = [s for s in required_skills if s in candidate_skills]
    missing = [s for s in required_skills if s not in candidate_skills]
    score = round((len(matched) / len(required_skills)) * 100, 2)
    return score, missing, matched


def _education_score(education_text: str) -> float:
    """
    Education is scored with a degree-level + field-relevance heuristic
    rather than raw TF-IDF cosine similarity. Education sections are short
    ("MS in Computer Science, XYZ University, 2019") and TF-IDF similarity
    against a job description tends to be near-zero regardless of quality,
    since degree titles rarely share vocabulary with a JD's requirements
    list. A rule-based score is more reliable here.
    """
    if not education_text.strip():
        return 0.0
    text_lower = education_text.lower()

    if any(k in text_lower for k in ["phd", "doctorate"]):
        base = 100.0
    elif any(k in text_lower for k in ["master", "ms ", "m.s", "msc"]):
        base = 90.0
    elif any(k in text_lower for k in ["bachelor", "bs ", "b.s", "bsc"]):
        base = 75.0
    else:
        base = 40.0  # some education listed, degree level unclear

    if any(field in text_lower for field in RELEVANT_FIELDS):
        base = min(base + 10.0, 100.0)

    return round(base, 2)


def _certification_score(cert_text: str, matched_skills: list, required_skills: list) -> float:
    """Certifications are scored on: (a) how many were listed, and (b) how
    many mention a skill that's actually relevant to this job."""
    if not cert_text.strip():
        return 0.0
    cert_lower = cert_text.lower()
    relevant_hits = sum(1 for s in required_skills if s.lower() in cert_lower)
    volume_score = min(len(cert_text.splitlines()) * 20, 60)
    relevance_score = min(relevant_hits * 20, 40)
    return round(min(volume_score + relevance_score, 100), 2)


def score_candidate(candidate: dict, job_description: str,
                     vector_store: ResumeVectorStore,
                     experience_similarity: float,
                     project_similarity: float) -> dict:
    """
    Computes the full weighted breakdown for one already-parsed candidate.
    experience_similarity / project_similarity are TF-IDF cosine scores
    (0-100, already scaled) precomputed in bulk by the caller (more
    efficient than re-fitting a TF-IDF space per candidate per section).
    """
    required_skills = _extract_required_skills(job_description)
    skill_score, missing_skills, matched_skills = _skill_score(
        candidate["skills"], required_skills
    )

    education_score = _education_score(candidate["education"])
    certification_score = _certification_score(
        candidate["certifications"], matched_skills, required_skills
    )

    breakdown = {
        "skills": round(skill_score, 2),
        "experience": round(experience_similarity, 2),
        "education": round(education_score, 2),
        "certifications": round(certification_score, 2),
        "projects": round(project_similarity, 2),
    }

    overall = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)

    strengths = [cat.title() for cat, val in breakdown.items() if val >= 80]
    weaknesses = [cat.title() for cat, val in breakdown.items() if val < 60]

    return {
        "name": candidate["name"],
        "filename": candidate["filename"],
        "email": candidate["email"],
        "phone": candidate["phone"],
        "overall_score": round(overall, 2),
        "breakdown": breakdown,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "all_skills": candidate["skills"],
        "strengths": strengths if strengths else ["No standout category yet"],
        "weaknesses": weaknesses if weaknesses else ["No major gaps found"],
        "education": candidate["education"],
        "experience": candidate["experience"],
        "certifications": candidate["certifications"],
        "projects": candidate["projects"],
    }


def rank_candidates(candidates: list, job_description: str) -> list:
    """
    Full pipeline: takes a list of parsed candidate dicts (from
    resume_parser.parse_resume) + a job description string, and returns
    them sorted best-to-worst with full score breakdowns attached.
    """
    if not candidates:
        return []

    def _weighted_doc(section_text: str, raw_text: str) -> str:
        # Repeat the section text so it dominates the TF-IDF weighting,
        # while still pulling in the rest of the resume's vocabulary for
        # extra overlap with the job description (see notebook.ipynb).
        section_text = section_text or ""
        return (section_text + " ") * 2 + raw_text

    exp_docs = [_weighted_doc(c["experience"], c["raw_text"]) for c in candidates]
    proj_docs = [_weighted_doc(c["projects"], c["raw_text"]) for c in candidates]

    raw_exp_scores = ResumeVectorStore().fit_and_score(job_description, exp_docs)
    raw_proj_scores = ResumeVectorStore().fit_and_score(job_description, proj_docs)

    exp_scores = [min(s * SEMANTIC_SCALE_FACTOR, 100) for s in raw_exp_scores]
    proj_scores = [min(s * SEMANTIC_SCALE_FACTOR, 100) for s in raw_proj_scores]

    results = []
    vs = ResumeVectorStore()
    for i, candidate in enumerate(candidates):
        result = score_candidate(
            candidate, job_description, vs,
            experience_similarity=exp_scores[i],
            project_similarity=proj_scores[i],
        )
        results.append(result)

    results.sort(key=lambda r: r["overall_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank
    return results


def score_tier(score: float) -> str:
    """Green / Yellow / Red tier used for the UI colour coding."""
    if score >= 80:
        return "green"
    if score >= 60:
        return "yellow"
    return "red"
