# 🧭 TalentLens — AI-Powered Resume Screening & Candidate Ranking System

> **Teerop Technologies — ML & AI Internship, Task 2**
> Upload resumes, drop in a job description, and get explainable, weighted
> candidate rankings — plus an AI chatbot that can answer questions about your
> candidates in plain English.

---

## 📚 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [Features](#-features)
3. [Architecture & Data Flow](#-architecture--data-flow)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [How Each File Works](#-how-each-file-works)
7. [Scoring Methodology](#-scoring-methodology)
8. [Getting Started (Run Locally)](#-getting-started-run-locally)
9. [Deploying to Streamlit Community Cloud](#-deploying-to-streamlit-community-cloud)
10. [Configuring Your Groq API Key Safely](#-configuring-your-groq-api-key-safely)
11. [Using the App](#-using-the-app)
12. [Optional: The EDA / Model Notebook](#-optional-the-eda--model-notebook)
13. [Known Limitations & Ideas for Later](#-known-limitations--ideas-for-later)

---

## 🎯 What This Project Does

Recruiters get dozens of resumes for one job opening and no easy way to compare
them objectively. **TalentLens** takes a stack of PDF resumes and a job
description, and automatically:

- Extracts each candidate's contact info, skills, education, experience,
  certifications, and projects.
- Scores every candidate against the job description across **5 weighted
  categories**.
- Ranks them from best to worst fit, with a clear reason for every score.
- Lets you ask an AI chatbot natural-language questions like *"Who has AWS
  experience?"* or *"Compare the top 2 candidates."*

---

## ✨ Features

| Feature | Description |
|---|---|
| 📎 **Multi-resume upload** | Upload any number of PDF resumes at once |
| 🧪 **One-click sample data** | Generates two ready-made demo resumes (AI/ML Engineer + Full-Stack Developer) so you can try the app instantly, no real resumes needed |
| 🔍 **Structured extraction** | Name, email, phone, 60+ recognised technical skills, education, experience, certifications, projects |
| 📝 **Flexible job description input** | Paste text or upload a JD as a PDF |
| ⚖️ **5-category weighted scoring** | Skills 35% · Experience 30% · Education 15% · Certifications 10% · Projects 10% |
| 🧠 **Semantic matching** | TF-IDF vectorization + cosine similarity, not just keyword matching |
| 🟢🟡🔴 **Colour-coded rankings** | Green ≥80%, Yellow 60–79%, Red <60% |
| 📊 **Interactive charts** | Grouped bar chart comparing every candidate across all 5 categories |
| 🕸️ **Radar charts** | Visual per-candidate score "fingerprint" |
| ⚖️ **Side-by-side compare tool** | Pick any two candidates and compare their breakdowns directly |
| ⬇️ **CSV export** | Download the full ranking table |
| 🤖 **AI HR chatbot** | Groq-hosted Llama 3.1 8B Instant, answers questions using the actual ranking data as context |
| 🎨 **Custom dark dashboard UI** | Built with hand-styled CSS on top of Streamlit — not the default theme |

---

## 🏗 Architecture & Data Flow

```
                    ┌─────────────────────────┐
                    │        Sidebar (UI)      │
                    │  API key · sample data    │
                    │  resume upload · JD input │
                    └────────────┬─────────────┘
                                 │  "Process & Rank" clicked
                                 ▼
                    ┌─────────────────────────┐
   PDF resumes ───▶ │   resume_parser.py       │  PyPDF2 text extraction
                    │   (regex + skill scan)   │  + regex/section parsing
                    └────────────┬─────────────┘
                                 │  structured candidate dicts
                                 ▼
                    ┌─────────────────────────┐
   Job description ▶│   vector_store.py        │  TF-IDF vectorization
                    │   (TF-IDF + cosine sim)  │  + cosine similarity
                    └────────────┬─────────────┘
                                 │  similarity scores
                                 ▼
                    ┌─────────────────────────┐
                    │   ranking_engine.py       │  5-category weighted
                    │   (weighted scoring)      │  scoring + strengths/
                    │                           │  weaknesses analysis
                    └────────────┬─────────────┘
                                 │  ranked candidate list
                                 ▼
                    ┌─────────────────────────┐
                    │        app.py (UI)        │  Rankings tab
                    │  Streamlit tabs + charts  │  Profiles tab
                    └────────────┬─────────────┘  Chatbot tab
                                 │  ranked list + JD as context
                                 ▼
                    ┌─────────────────────────┐
                    │      chatbot.py           │  Groq API call
                    │  (Groq / Llama 3.1 8B)    │  llama-3.1-8b-instant
                    └─────────────────────────┘
```

**In plain words:** the sidebar collects your inputs → `resume_parser.py` turns
raw PDFs into structured data → `vector_store.py` measures how semantically
close each resume section is to the job description → `ranking_engine.py`
combines that with skill/education/certification checks into one overall score
→ `app.py` renders everything in three tabs → `chatbot.py` lets you ask an AI
follow-up questions using that same ranked data as its only source of truth.

---

## 🛠 Tech Stack

- **Frontend / App:** Streamlit
- **PDF Processing:** PyPDF2
- **ML / NLP:** Scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- **Data Handling:** Pandas, NumPy
- **AI Chatbot:** Groq API — `llama-3.1-8b-instant`
- **Sample Resume Generation:** ReportLab (builds real PDF files on the fly)
- **Charts:** Plotly (bar charts + radar charts)
- No PyTorch, no FAISS, no GPU required — light enough to run anywhere,
  including free-tier cloud hosting.

---

## 📂 Project Structure

```
resume_screening_system/
├── app.py                      # Main Streamlit app (UI + tabs)
├── resume_parser.py            # PDF text extraction & structured parsing
├── ranking_engine.py           # 5-category weighted scoring system
├── vector_store.py             # TF-IDF vectorization + cosine similarity
├── chatbot.py                  # Groq API (Llama 3.1 8B Instant) integration
├── cv_generator.py             # Generates 2 sample PDF resumes for demos
├── notebook.ipynb              # Optional: EDA + model comparison notebook
├── requirements.txt            # Exact, tested dependency versions
├── .gitignore                  # Keeps secrets & junk out of git
├── .streamlit/
│   └── secrets.toml.example    # Template for your Groq API key (copy, don't commit)
└── README.md                   # You are here
```

---

## 🧩 How Each File Works

**`resume_parser.py`**
Reads a PDF with PyPDF2, then uses regex to pull out an email, a phone number,
and the first short line as the candidate's name. A list of 60+ technology
keywords (Python, TensorFlow, AWS, Docker, React, …) is scanned against the
resume text to detect skills. Section headers like `EDUCATION`, `EXPERIENCE`,
`CERTIFICATIONS`, `PROJECTS` are used to slice the resume into the relevant
blocks of text.

**`vector_store.py`**
Wraps Scikit-learn's `TfidfVectorizer`. Fits one shared vector space over the
job description + every resume section, then computes cosine similarity so
resumes that are *semantically* close to the JD score well — even if they
don't use the exact same words.

**`ranking_engine.py`**
Combines everything into the final weighted score:
- **Skills (35%)** — % of the JD's required skills the candidate actually has.
- **Experience (30%)** — TF-IDF/cosine similarity between the experience
  section and the JD, scaled up for readability (see code comments for why).
- **Education (15%)** — degree-level + relevant-field heuristic (Bachelor's /
  Master's / PhD in a technical field).
- **Certifications (10%)** — how many certifications are listed and how many
  are relevant to the required skills.
- **Projects (10%)** — TF-IDF/cosine similarity between listed projects and
  the JD.

It also flags **strengths** (any category ≥80%) and **weaknesses** (any
category <60%), and lists exactly which required skills are missing.

**`chatbot.py`**
Builds a system prompt out of the *current* ranking results (names, scores,
strengths/weaknesses, missing skills) and sends it to Groq's
`llama-3.1-8b-instant` model at `temperature=0.5` alongside the recruiter's
question, so answers are grounded in the real scoring data instead of the
model guessing.

**`cv_generator.py`**
Builds two realistic sample resumes as actual PDF files (using ReportLab) so
you can demo the whole app in one click without needing real resumes on hand.

**`app.py`**
Ties everything together: sidebar inputs → processing pipeline → three tabs
(Rankings, Candidate Profiles, AI Chatbot), all styled with custom CSS.

---

## 📐 Scoring Methodology

```
Overall Score =  (Skill Match        × 0.35)
              +  (Experience Match   × 0.30)
              +  (Education Match    × 0.15)
              +  (Certification Match× 0.10)
              +  (Project Relevance  × 0.10)
```

Every sub-score is on a 0–100 scale, so the overall score is too.

| Tier | Score Range | Meaning |
|---|---|---|
| 🟢 Green | ≥ 80% | Excellent Match |
| 🟡 Yellow | 60–79% | Good Match |
| 🔴 Red | < 60% | Needs Improvement |

---

## 🚀 Getting Started (Run Locally)

```bash
# 1. Clone your repo
git clone https://github.com/<your-username>/<your-repo>.git
cd resume_screening_system

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (see next section)

# 5. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, pick your repo, branch, and set the main file path to
   `app.py`.
4. Before deploying, open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_actual_groq_key_here"
   ```
5. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` automatically — the versions in this repo have already
   been tested end-to-end so there shouldn't be any dependency surprises.

The app also works with **zero configuration**: if no secret/env var is set,
the sidebar simply asks the user to paste their own Groq API key at runtime —
handy if you want to share the deployed link publicly without exposing your
own key.

---

## 🔑 Configuring Your Groq API Key Safely

**Never hardcode your API key inside any `.py` file, and never commit it to
GitHub.** This project reads it in this order:

1. `GROQ_API_KEY` environment variable (useful for Render, Docker, etc.)
2. `st.secrets["GROQ_API_KEY"]` (Streamlit Cloud's secrets manager)
3. Manually typed into the sidebar's password field (works for a quick local test)

**Local setup:**
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key
```
`.streamlit/secrets.toml` is already listed in `.gitignore`, so it will never
be committed by accident.

Get a free key at **[console.groq.com/keys](https://console.groq.com/keys)**.

> ⚠️ If a Groq key was ever pasted into a chat, notebook, screenshot, or commit
> by mistake, treat it as compromised and generate a new one from the Groq
> console — keys are free and instant to rotate.

---

## 🖱 Using the App

1. Open the app and paste your Groq API key in the sidebar (only needed for
   the chatbot tab).
2. Click **"✨ Generate Sample Resumes"** to try it instantly, or upload your
   own PDF resumes.
3. Paste a job description, or upload one as a PDF.
4. Click **"🚀 Process & Rank Candidates"**.
5. Explore the **Rankings** tab (table, charts, compare tool), the
   **Candidate Profiles** tab (full parsed detail), and the **AI Chatbot** tab.

---

## 📓 Optional: The EDA / Model Notebook

`notebook.ipynb` is a bonus, fully-executed notebook that documents the data
science behind the scoring system: a small labelled synthetic dataset for
testing, a comparison of different vectorization techniques, how the
similarity scaling was calibrated, a ranking-accuracy evaluation, and a
performance benchmark. It's not required to run the app — the app is fully
self-contained in the `.py` files above — but it's there if you want to see
the reasoning and experiments behind the scoring design, or want something to
show during a technical review.

```bash
pip install jupyter
jupyter notebook notebook.ipynb
```

---

## ⚠️ Known Limitations & Ideas for Later

- Extraction is regex/keyword-based, so heavily templated or image-based
  (scanned) PDF resumes may parse less reliably than plain-text PDFs.
- Skill detection is limited to the 60+ keyword list in `resume_parser.py` —
  easy to extend by adding more entries to `SKILL_CATEGORIES`.
- Education scoring uses a degree-level heuristic rather than deep semantic
  understanding of transcripts.
- Future upgrade path: sentence-embedding models (e.g. `sentence-transformers`)
  for even better semantic matching, and LLM-based structured extraction for
  messier real-world resume formats.

---

Built for the Teerop Technologies ML & AI Internship — Task 2.
