"""
app.py
------
Main Streamlit application for the AI-Powered Resume Screening &
Candidate Ranking System.

Layout:
    Sidebar  -> Step-by-step guide, Groq API key (auto-loaded from secrets),
                sample data, resume upload, job description, Process button
    Tab 1    -> Rankings (table, charts, strengths/weaknesses, compare)
    Tab 2    -> Candidate Profiles (full parsed data)
    Tab 3    -> AI Chatbot (Groq Llama 3.1)

Run:   streamlit run app.py
Deploy: push to GitHub -> Streamlit Cloud -> set GROQ_API_KEY in Secrets
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load local .env if present (for development)
load_dotenv()

# Import project modules
from resume_parser import parse_resume, extract_text_from_pdf
from ranking_engine import rank_candidates, score_tier
from chatbot import ask_chatbot, QUICK_QUESTIONS
from cv_generator import SAMPLE_CV_GENERATORS

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentLens – AI Resume Screening",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_JD = """Senior AI/ML Engineer

Requirements: 5+ years Python and ML experience, TensorFlow/PyTorch, NLP, \
Computer Vision, AWS/GCP, Docker, Kubernetes, MLOps. Preferred: RAG systems, \
FastAPI, PostgreSQL, Team leadership."""

TIER_COLORS = {
    "green": "#1FB88E",
    "yellow": "#E8A93B",
    "red": "#E4574C",
}
TIER_LABELS = {"green": "Excellent Match", "yellow": "Good Match", "red": "Needs Improvement"}


# ---------------------------------------------------------------------------
# Custom CSS (Enhanced)
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
        /* -------- Fonts -------- */
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        h1, h2, h3, .hero-title { font-family: 'Sora', sans-serif; }

        /* -------- Background -------- */
        .stApp {
            background: radial-gradient(circle at 15% 0%, #17233A 0%, #0B1220 45%, #0B1220 100%);
            color: #E7ECF5;
        }

        /* -------- Sidebar -------- */
        section[data-testid="stSidebar"] {
            background: #101A2E;
            border-right: 1px solid #223050;
        }
        section[data-testid="stSidebar"] * { color: #CBD5E8 !important; }

        /* -------- Hero Banner -------- */
        .hero-banner {
            background: linear-gradient(120deg, #123B36 0%, #0F2C4C 60%, #0B1220 100%);
            border: 1px solid #1F4A6B;
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 22px;
        }
        .hero-title {
            font-size: 30px; font-weight: 800; color: #F4F8FF; margin-bottom: 4px;
        }
        .hero-sub {
            color: #9FB3D1; font-size: 15px;
        }
        .eyebrow {
            display: inline-block; background: rgba(31,184,142,0.15); color: #3FE0B4;
            border: 1px solid rgba(63,224,180,0.35); border-radius: 999px;
            padding: 3px 12px; font-size: 12px; font-weight: 600; letter-spacing: .06em;
            text-transform: uppercase; margin-bottom: 10px;
        }

        /* -------- Cards -------- */
        .metric-card {
            background: #121C31; border: 1px solid #223050; border-radius: 14px;
            padding: 16px 18px;
        }
        .candidate-card {
            background: #121C31; border: 1px solid #223050; border-radius: 16px;
            padding: 20px 22px; margin-bottom: 16px;
        }
        .rank-badge {
            display: inline-flex; align-items: center; justify-content: center;
            width: 34px; height: 34px; border-radius: 50%;
            background: #1B2947; color: #E7ECF5; font-weight: 700; font-size: 14px;
            border: 1px solid #2E4470;
        }
        .score-pill {
            display: inline-block; padding: 4px 14px; border-radius: 999px;
            font-weight: 700; font-size: 14px; color: #0B1220;
        }
        .tag {
            display: inline-block; background: #1B2947; color: #A9C4FF;
            border: 1px solid #2E4470; border-radius: 8px; padding: 2px 9px;
            font-size: 12px; margin: 2px 4px 2px 0;
        }
        .tag-missing {
            background: rgba(228,87,76,0.12); color: #F1948A; border: 1px solid rgba(228,87,76,0.4);
        }
        .section-label {
            color: #7C93BF; font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
            font-weight: 700; margin-top: 14px; margin-bottom: 4px;
        }

        /* -------- Buttons -------- */
        .stButton>button {
            background: linear-gradient(120deg, #1FB88E, #159C86);
            color: #06140F; font-weight: 700; border: none; border-radius: 10px;
            padding: 0.55em 1.1em;
        }
        .stButton>button:hover {
            background: linear-gradient(120deg, #24D9A8, #1AB89E); color: #06140F;
        }

        /* -------- Tabs -------- */
        div[data-testid="stTabs"] button { font-weight: 600; }

        /* -------- Info/Warning boxes -------- */
        .stAlert { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: score pill HTML
# ---------------------------------------------------------------------------
def score_pill_html(score: float) -> str:
    tier = score_tier(score)
    color = TIER_COLORS[tier]
    return f'<span class="score-pill" style="background:{color};">{score:.1f}%</span>'


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in [
    ("candidates", []),
    ("results", []),
    ("job_description", DEFAULT_JD),
    ("chat_history", []),
    ("sample_files", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar – with clear step‑by‑step guide
# ---------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.markdown("### 🧭 TalentLens")
        st.caption("AI Resume Screening & Candidate Ranking")

        # ========== STEP 0: API Key (auto) ==========
        st.markdown("#### 🔑 API Key")
        # Try to get from secrets first, then environment
        api_key = None
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("✅ Key loaded from secrets", icon="🔒")
        except Exception:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                st.success("✅ Key loaded from environment", icon="🔒")
            else:
                st.warning("⚠️ No API key found. Please set GROQ_API_KEY in Streamlit Secrets or .env")
                api_key = st.text_input(
                    "Paste your Groq API key (optional if you only use ranking)",
                    type="password",
                    placeholder="gsk_...",
                    help="Get a free key at console.groq.com"
                )
                if api_key:
                    st.session_state.manual_api_key = api_key
                else:
                    api_key = None

        st.markdown("---")

        # ========== STEP 1: Try instantly ==========
        st.markdown("#### 🧪 1. Try it instantly")
        if st.button("✨ Generate Sample Resumes", use_container_width=True):
            st.session_state.sample_files = {
                label: fn() for label, fn in SAMPLE_CV_GENERATORS.items()
            }
            st.success(f"Generated {len(st.session_state.sample_files)} sample resumes below ⬇️")

        # Show sample downloads if any
        if st.session_state.sample_files:
            with st.expander("📁 Download generated resumes"):
                for label, pdf_bytes in st.session_state.sample_files.items():
                    st.download_button(
                        f"⬇️ {label}",
                        data=pdf_bytes,
                        file_name=label.split(" - ")[0].replace(" ", "_") + ".pdf",
                        mime="application/pdf",
                        key=f"dl_{label}"
                    )

        # ========== STEP 2: Upload resumes ==========
        st.markdown("#### 📄 2. Upload Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once."
        )

        include_samples = False
        if st.session_state.sample_files:
            include_samples = st.checkbox(
                f"Include {len(st.session_state.sample_files)} generated sample resume(s)",
                value=True,
                help="Check this to add the sample candidates to the ranking."
            )

        # ========== STEP 3: Job Description ==========
        st.markdown("#### 🎯 3. Job Description")
        jd_text = st.text_area(
            "Paste job description",
            value=st.session_state.job_description,
            height=150,
            help="Describe the role, required skills, experience, etc."
        )
        jd_pdf = st.file_uploader(
            "...or upload JD as PDF",
            type=["pdf"],
            key="jd_pdf",
            help="If you upload a PDF, its text will be used instead of the text above."
        )
        if jd_pdf is not None:
            jd_text = extract_text_from_pdf(jd_pdf)
            st.info("📄 Using text extracted from uploaded JD PDF.")

        # ========== STEP 4: Process ==========
        st.markdown("---")
        process = st.button(
            "🚀 4. Process & Rank Candidates",
            use_container_width=True,
            type="primary",
            help="Click to start parsing and ranking based on the uploaded resumes and job description."
        )

        # ========== Summary ==========
        total_files = (len(uploaded_resumes) if uploaded_resumes else 0) + (
            len(st.session_state.sample_files) if include_samples else 0
        )
        st.caption(f"📎 {total_files} resume(s) ready to process")

        # ========== Quick guide ==========
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Add your Groq API key** (if you want the AI chatbot).  
            2. **Generate sample resumes** or upload your own PDFs.  
            3. **Provide a job description** (paste or upload PDF).  
            4. **Click "Process & Rank Candidates"** to see results.  
            5. Explore the **Rankings**, **Profiles**, and **AI Chatbot** tabs.
            """)

        return uploaded_resumes, include_samples, jd_text, process, api_key


# ---------------------------------------------------------------------------
# Processing function (with progress feedback)
# ---------------------------------------------------------------------------
def process_resumes(uploaded_resumes, include_samples, jd_text):
    candidates = []
    errors = []

    files_to_process = []
    if uploaded_resumes:
        for f in uploaded_resumes:
            files_to_process.append((f, f.name))
    if include_samples:
        for label, pdf_bytes in st.session_state.sample_files.items():
            files_to_process.append((io.BytesIO(pdf_bytes), label))

    if not files_to_process:
        st.warning("Please upload at least one resume, or generate sample resumes first.")
        return

    progress = st.progress(0, text="Parsing resumes...")
    for i, (file_obj, name) in enumerate(files_to_process):
        try:
            file_obj.seek(0)
            parsed = parse_resume(file_obj, name)
            candidates.append(parsed)
        except Exception as e:
            errors.append(f"{name}: {e}")
        progress.progress((i + 1) / len(files_to_process), text=f"Parsed {name}")
    progress.empty()

    if errors:
        st.error("Some files failed to parse:\n" + "\n".join(errors))

    if candidates:
        with st.spinner("Scoring candidates with TF-IDF + weighted matching..."):
            st.session_state.results = rank_candidates(candidates, jd_text)
            st.session_state.candidates = candidates
            st.session_state.job_description = jd_text
            st.session_state.chat_history = []  # reset chat
        st.success(f"✅ Ranked {len(candidates)} candidate(s) successfully.")
    else:
        st.error("No candidates could be parsed. Please check your resume files.")


# ---------------------------------------------------------------------------
# Tab 1: Rankings
# ---------------------------------------------------------------------------
def render_rankings_tab():
    results = st.session_state.results
    if not results:
        st.info("📭 Upload resumes and a job description, then click **Process & Rank Candidates** in the sidebar.")
        return

    # ---- Metric cards ----
    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-card"><div class="section-label">👥 Candidates</div>'
                      f'<h2>{len(results)}</h2></div>', unsafe_allow_html=True)
    top = results[0]
    cols[1].markdown(f'<div class="metric-card"><div class="section-label">🏆 Top Candidate</div>'
                      f'<h3>{top["name"]}</h3></div>', unsafe_allow_html=True)
    avg = sum(r["overall_score"] for r in results) / len(results)
    cols[2].markdown(f'<div class="metric-card"><div class="section-label">📊 Average Score</div>'
                      f'<h2>{avg:.1f}%</h2></div>', unsafe_allow_html=True)
    green_count = sum(1 for r in results if score_tier(r["overall_score"]) == "green")
    cols[3].markdown(f'<div class="metric-card"><div class="section-label">🌟 Excellent Matches</div>'
                      f'<h2>{green_count}</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Ranking table ----
    st.markdown("#### 🏆 Candidate Ranking Table")
    df_rows = []
    for r in results:
        df_rows.append({
            "Rank": r["rank"],
            "Candidate": r["name"],
            "Overall": r["overall_score"],
            "Skills": r["breakdown"]["skills"],
            "Experience": r["breakdown"]["experience"],
            "Education": r["breakdown"]["education"],
            "Certifications": r["breakdown"]["certifications"],
            "Projects": r["breakdown"]["projects"],
        })
    df = pd.DataFrame(df_rows)

    def highlight_tier(val):
        tier = score_tier(val)
        return f'background-color: {TIER_COLORS[tier]}33; color: #E7ECF5; font-weight:600;'

    styled = df.style.map(highlight_tier, subset=["Overall"]).format(
        {c: "{:.1f}%" for c in ["Overall", "Skills", "Experience", "Education", "Certifications", "Projects"]}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ---- Export ----
    st.download_button(
        "⬇️ Export Rankings as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="candidate_rankings.csv",
        mime="text/csv",
    )

    # ---- Score comparison bar chart ----
    st.markdown("#### 📊 Score Comparison Across Categories")
    fig = go.Figure()
    categories = ["Skills", "Experience", "Education", "Certifications", "Projects"]
    colors_seq = ["#3FE0B4", "#5B8DEF", "#E8A93B", "#B892FF", "#F1948A"]
    for cat, color in zip(categories, colors_seq):
        fig.add_trace(go.Bar(
            name=cat,
            x=[r["name"] for r in results],
            y=[r["breakdown"][cat.lower()] for r in results],
            marker_color=color,
        ))
    fig.update_layout(
        barmode="group",
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Strengths & Weaknesses ----
    st.markdown("#### 🔍 Strengths & Weaknesses")
    for r in results:
        tier = score_tier(r["overall_score"])
        with st.expander(f"#{r['rank']} — {r['name']} — {r['overall_score']:.1f}% ({TIER_LABELS[tier]})"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("**✅ Strengths**")
                for s in r["strengths"]:
                    st.markdown(f"- {s}")
            with col2:
                st.markdown("**⚠️ Weaknesses**")
                for w in r["weaknesses"]:
                    st.markdown(f"- {w}")
            with col3:
                st.plotly_chart(radar_chart([r]), use_container_width=True,
                                 key=f"radar_{r['rank']}_{r['name']}")
            if r["missing_skills"]:
                st.markdown("**Missing required skills:**")
                st.markdown("".join(f'<span class="tag tag-missing">{s}</span>' for s in r["missing_skills"]),
                             unsafe_allow_html=True)

    # ---- Compare section ----
    render_compare_section(results)


# ---------------------------------------------------------------------------
# Radar chart helper
# ---------------------------------------------------------------------------
def radar_chart(candidates_subset: list):
    categories = ["Skills", "Experience", "Education", "Certifications", "Projects"]
    palette = ["#3FE0B4", "#B892FF"]
    fig = go.Figure()
    for r, color in zip(candidates_subset, palette):
        values = [r["breakdown"][c.lower()] for c in categories]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=r["name"],
            line_color=color,
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#7C93BF"),
            bgcolor="rgba(0,0,0,0)"
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=len(candidates_subset) > 1,
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Compare two candidates
# ---------------------------------------------------------------------------
def render_compare_section(results):
    st.markdown("#### ⚖️ Compare Two Candidates")
    if len(results) < 2:
        st.caption("Add at least 2 candidates to unlock side-by-side comparison.")
        return
    names = [r["name"] for r in results]
    col1, col2 = st.columns(2)
    name_a = col1.selectbox("Candidate A", names, index=0, key="compare_a")
    name_b = col2.selectbox("Candidate B", names, index=min(1, len(names) - 1), key="compare_b")

    cand_a = next(r for r in results if r["name"] == name_a)
    cand_b = next(r for r in results if r["name"] == name_b)

    cc1, cc2 = st.columns(2)
    cc1.markdown(f'<div class="metric-card"><div class="section-label">{name_a}</div>'
                 f'{score_pill_html(cand_a["overall_score"])}</div>', unsafe_allow_html=True)
    cc2.markdown(f'<div class="metric-card"><div class="section-label">{name_b}</div>'
                 f'{score_pill_html(cand_b["overall_score"])}</div>', unsafe_allow_html=True)

    st.plotly_chart(radar_chart([cand_a, cand_b]), use_container_width=True, key="compare_radar")

    compare_df = pd.DataFrame({
        "Category": ["Skills", "Experience", "Education", "Certifications", "Projects"],
        name_a: [cand_a["breakdown"][c.lower()] for c in ["skills", "experience", "education", "certifications", "projects"]],
        name_b: [cand_b["breakdown"][c.lower()] for c in ["skills", "experience", "education", "certifications", "projects"]],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: Candidate Profiles
# ---------------------------------------------------------------------------
def render_profiles_tab():
    results = st.session_state.results
    if not results:
        st.info("📭 Process resumes first to see detailed candidate profiles here.")
        return

    for r in results:
        tier = score_tier(r["overall_score"])
        st.markdown(f"""
        <div class="candidate-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="rank-badge">#{r['rank']}</span>
                    &nbsp; <span style="font-size:20px; font-weight:700;">{r['name']}</span>
                </div>
                {score_pill_html(r['overall_score'])}
            </div>
            <div style="color:#7C93BF; font-size:13px; margin-top:6px;">
                ✉️ {r['email']} &nbsp;|&nbsp; 📞 {r['phone']}
            </div>
            <div class="section-label">🛠️ Skills</div>
            <div>{"".join(f'<span class="tag">{s}</span>' for s in r['all_skills']) or "No skills detected"}</div>
            <div class="section-label">🎓 Education</div>
            <div style="font-size:14px; color:#CBD5E8;">{r['education'] or "Not found"}</div>
            <div class="section-label">💼 Experience</div>
            <div style="font-size:14px; color:#CBD5E8; white-space:pre-line;">{r['experience'] or "Not found"}</div>
            <div class="section-label">📜 Certifications</div>
            <div style="font-size:14px; color:#CBD5E8; white-space:pre-line;">{r['certifications'] or "None listed"}</div>
            <div class="section-label">📁 Projects</div>
            <div style="font-size:14px; color:#CBD5E8; white-space:pre-line;">{r['projects'] or "None listed"}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3: AI Chatbot
# ---------------------------------------------------------------------------
def render_chatbot_tab(api_key):
    results = st.session_state.results
    if not results:
        st.info("📭 Process resumes first, then come back here to ask the AI chatbot about your candidates.")
        return
    if not api_key:
        st.warning("🔑 Please enter your Groq API key in the sidebar to enable the chatbot.")
        return

    st.markdown("#### 💬 Ask about your candidates")
    st.caption("Powered by Groq · Llama 3.1 8B Instant")

    # Quick question buttons
    qcols = st.columns(len(QUICK_QUESTIONS))
    for i, q in enumerate(QUICK_QUESTIONS):
        if qcols[i].button(q, use_container_width=True, key=f"quick_{i}"):
            st.session_state["pending_question"] = q

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about the candidates...") or pending

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_chatbot(
                        api_key,
                        user_input,
                        results,
                        st.session_state.job_description,
                        st.session_state.chat_history[:-1],
                    )
                except Exception as e:
                    answer = f"⚠️ Chatbot error: {e}"
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()

    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <div class="eyebrow">AI/ML Internship · Task 2</div>
        <div class="hero-title">🧭 TalentLens — AI Resume Screening &amp; Candidate Ranking</div>
        <div class="hero-sub">Upload resumes, drop in a job description, and get explainable,
        weighted candidate rankings powered by TF-IDF matching + a Groq-powered HR chatbot.</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    uploaded_resumes, include_samples, jd_text, process, api_key = sidebar()

    # Process
    if process:
        process_resumes(uploaded_resumes, include_samples, jd_text)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "👤 Candidate Profiles", "🤖 AI Chatbot"])
    with tab1:
        render_rankings_tab()
    with tab2:
        render_profiles_tab()
    with tab3:
        render_chatbot_tab(api_key)


if __name__ == "__main__":
    main()