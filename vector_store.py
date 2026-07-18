"""
vector_store.py
----------------
A tiny, dependency-light "vector store" built on Scikit-learn's TF-IDF
vectorizer. This is what gives the system semantic matching instead of
plain keyword search: two documents that use different words for the
same idea ("built REST APIs" vs "developed backend services") still end
up with a non-zero cosine similarity because of shared/related terms.

Why TF-IDF + cosine similarity instead of embeddings/FAISS?
  - Zero GPU requirement, no heavy downloads, works instantly on any
    machine (Streamlit Cloud included) - exactly what the brief asks for.
  - It's fast enough to score dozens of resumes against a JD in well
    under a second.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeVectorStore:
    """Fits one TF-IDF space over [job_description, resume_1, ..., resume_n]
    so every document is comparable in the same vector space, then exposes
    cosine similarity scores of each resume against the job description."""

    def __init__(self, ngram_range=(1, 2), max_features=5000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self._fitted = False

    def fit_and_score(self, job_description: str, resume_texts: list) -> list:
        """
        Returns a list of similarity scores (0-100 scale), one per resume,
        in the same order as resume_texts.
        """
        documents = [job_description] + resume_texts
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self._fitted = True

        jd_vector = tfidf_matrix[0:1]
        resume_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(jd_vector, resume_vectors)[0]
        return [round(float(s) * 100, 2) for s in similarities]

    def top_terms(self, job_description: str, top_n: int = 15) -> list:
        """Utility used by the EDA notebook: which terms in the JD carry the
        most TF-IDF weight (i.e. what the matcher is actually keying on)."""
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vec.fit_transform([job_description])
        scores = matrix.toarray()[0]
        terms = vec.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
