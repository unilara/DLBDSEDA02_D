import re
import pandas as pd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


# ---------- Helpers ----------

def normalize_text(text: str) -> str:
    """Light normalization before spaCy: lowercase + remove URLs."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return text


def spacy_clean_texts(nlp, texts):
    """
    Clean texts with spaCy:
    - tokenize
    - remove stopwords
    - keep alphabetic tokens
    - lemmatize
    - remove short tokens
    """
    cleaned = []
    # nlp.pipe is much faster than calling nlp() in a loop
    for doc in nlp.pipe(texts, batch_size=500):
        tokens = []
        for token in doc:
            if token.is_stop:
                continue
            if not token.is_alpha:
                continue
            lemma = token.lemma_.lower().strip()
            if len(lemma) <= 2:
                continue
            if lemma in STOP_WORDS:
                continue
            # Optional: remove placeholder-like tokens
            # (keeps your earlier results comparable)
            # if lemma == "xxxx":
            #     continue
            tokens.append(lemma)
        cleaned.append(" ".join(tokens))
    return cleaned


def print_top_words(model, feature_names, n_top_words: int = 10):
    """Print top words per topic for sklearn LDA/NMF."""
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[:-n_top_words - 1:-1]
        words = [feature_names[i] for i in top_idx]
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join(words))


def compute_coherence_values(dictionary, corpus, texts, start=2, limit=10, passes=5):
    """Compute c_v coherence for different topic counts using gensim LDA."""
    coherence_values = []
    for num_topics in range(start, limit):
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes
        )
        # processes=1 avoids multiprocessing issues on macOS/Python 3.12
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1
        )
        coherence_values.append(coherence_model.get_coherence())
    return coherence_values


# ---------- Main ----------

def main():
    print("START")

    # 1) Load data
    df = pd.read_csv("customercomplaints.csv", low_memory=False)
    texts_all = df["Consumer complaint narrative"].dropna().astype(str)
    print("Anzahl Texte gesamt:", len(texts_all))

    # 2) Load spaCy model (for tokenization + lemmatization)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' ist nicht installiert.\n"
            "Bitte im aktivierten venv ausführen:\n"
            "  python -m spacy download en_core_web_sm\n"
        )

    # ==========================================================
    # A) Main analysis sample (quality)
    # ==========================================================
    texts_main = texts_all.sample(n=20000, random_state=42)
    print("Sample Größe (Hauptanalyse):", len(texts_main))

    # Normalize then spaCy-clean
    texts_main_norm = [normalize_text(t) for t in texts_main.tolist()]
    cleaned_main_list = spacy_clean_texts(nlp, texts_main_norm)

    # show example
    print("\nBeispiel nachher (cleaned):")
    print(cleaned_main_list[0][:250])

    # 3) Vectorization (two techniques)
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=10)
    count_matrix = count_vectorizer.fit_transform(cleaned_main_list)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_main_list)

    print("\nVektorisierung abgeschlossen.")
    print("Count-Matrix Shape:", count_matrix.shape)
    print("TFIDF-Matrix Shape:", tfidf_matrix.shape)

    # 4) Topic models (initial example with k=5)
    n_topics_initial = 5

    lda = LatentDirichletAllocation(n_components=n_topics_initial, random_state=42)
    lda.fit(count_matrix)

    nmf = NMF(n_components=n_topics_initial, random_state=42)
    nmf.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={n_topics_initial}) ---")
    print_top_words(lda, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={n_topics_initial}) ---")
    print_top_words(nmf, tfidf_vectorizer.get_feature_names_out())

    # ==========================================================
    # B) Coherence score sample (speed)
    # ==========================================================
    print("\nBerechne Coherence Scores (schnelleres Sample)...")

    texts_coh = texts_all.sample(n=5000, random_state=42)
    print("Sample Größe (Coherence):", len(texts_coh))

    texts_coh_norm = [normalize_text(t) for t in texts_coh.tolist()]
    cleaned_coh_list = spacy_clean_texts(nlp, texts_coh_norm)

    tokenized_coh = [t.split() for t in cleaned_coh_list]

    dictionary = Dictionary(tokenized_coh)
    corpus = [dictionary.doc2bow(t) for t in tokenized_coh]

    coherence_values = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        texts=tokenized_coh,
        start=2,
        limit=10,
        passes=5
    )

    for k, score in enumerate(coherence_values, start=2):
        print(f"Topics = {k}, Coherence Score = {score:.4f}")

    best_k = coherence_values.index(max(coherence_values)) + 2
    print("\nOptimale Anzahl von Topics laut Coherence:", best_k)

    # 5) Final topic models with best_k
    print("\nTrainiere finale Modelle mit optimalem k...")

    lda_final = LatentDirichletAllocation(n_components=best_k, random_state=42)
    lda_final.fit(count_matrix)

    nmf_final = NMF(n_components=best_k, random_state=42)
    nmf_final.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={best_k}) ---")
    print_top_words(lda_final, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={best_k}) ---")
    print_top_words(nmf_final, tfidf_vectorizer.get_feature_names_out())

    print("ENDE")


if __name__ == "__main__":
    main()
