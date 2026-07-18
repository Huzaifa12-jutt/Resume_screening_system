"""
chatbot.py
----------
Thin wrapper around the Groq API (Llama 3.1 8B Instant) that turns the
ranked candidate list into context an HR user can ask natural-language
questions about ("Who has AWS experience?", "Compare John and Sarah").

Design notes:
    - temperature=0.5 as specified in the brief (consistent, still natural)
    - the system prompt is rebuilt every call with the *current* ranking
      results, so the chatbot always answers from the latest data, not a
      stale snapshot.
    - Chat history is passed in/out as a plain list of {role, content}
      dicts so app.py can keep it in st.session_state without this module
      needing to know anything about Streamlit.
"""

from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"


def _build_context(ranked_candidates: list, job_description: str) -> str:
    lines = [
        "You are an HR assistant helping a recruiter evaluate candidates "
        "for the following role:",
        f"JOB DESCRIPTION:\n{job_description}\n",
        "CANDIDATE RANKINGS (highest score first):",
    ]
    for c in ranked_candidates:
        lines.append(
            f"\n#{c['rank']} {c['name']} - Overall Score: {c['overall_score']}%\n"
            f"  Skills matched: {', '.join(c['matched_skills']) or 'None'}\n"
            f"  Missing skills: {', '.join(c['missing_skills']) or 'None'}\n"
            f"  Breakdown -> Skills: {c['breakdown']['skills']}%, "
            f"Experience: {c['breakdown']['experience']}%, "
            f"Education: {c['breakdown']['education']}%, "
            f"Certifications: {c['breakdown']['certifications']}%, "
            f"Projects: {c['breakdown']['projects']}%\n"
            f"  Strengths: {', '.join(c['strengths'])}\n"
            f"  Weaknesses: {', '.join(c['weaknesses'])}"
        )
    lines.append(
        "\nAnswer the recruiter's questions using only the information "
        "above. Be concise, specific, and reference candidate names and "
        "numbers directly. If asked to recommend a candidate, justify it "
        "using the score breakdown."
    )
    return "\n".join(lines)


def ask_chatbot(api_key: str, user_message: str, ranked_candidates: list,
                 job_description: str, chat_history: list) -> str:
    """
    api_key         : the user's Groq API key (never hardcoded/stored)
    user_message    : the new question from the recruiter
    ranked_candidates: output of ranking_engine.rank_candidates(...)
    job_description : the JD text currently in use
    chat_history    : list of {"role": "user"/"assistant", "content": str}
                       from earlier turns in this session (excludes system)
    """
    client = Groq(api_key=api_key)

    system_prompt = _build_context(ranked_candidates, job_description)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.5,
        max_tokens=600,
    )
    return response.choices[0].message.content


QUICK_QUESTIONS = [
    "Who is the best candidate for this role and why?",
    "Compare the top 2 candidates side by side.",
    "Which candidates are missing critical required skills?",
    "Who has the strongest hands-on project experience?",
]
