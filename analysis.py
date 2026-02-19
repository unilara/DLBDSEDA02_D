import re
import numpy as np
import pandas as pd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import gensim
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Globale Einstellungen für Reproduzierbarkeit
SEED = 42
np.random.seed(SEED)

# Hilfsfunktionen
def normalize_text(text: str) -> str:
    """Leichte Vor-Normalisierung: Kleinschreibung + Entfernen von URLs."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return text


def spacy_clean_texts(nlp, texts):
    """
    Textvorverarbeitung mit spaCy:
    - Tokenisierung
    - Stopwörter entfernen
    - nur alphabetische Tokens behalten (keine Zahlen/Sonderzeichen)
    - Lemmatisierung
    - kurze Tokens entfernen (<=2 Zeichen)
    Ergebnis: Liste aus 'sauberen Texten' (Strings), geeignet für Vectorizer.
    """
    cleaned = []

    # nlp.pipe ist deutlich schneller als nlp() pro Text
    for doc in nlp.pipe(texts, batch_size=200):
        tokens = []
        for token in doc:
            # Stopwörter raus
            if token.is_stop:
                continue

            # Nur Buchstaben
            if not token.is_alpha:
                continue

            lemma = token.lemma_.lower().strip()

            # Sehr kurze Tokens entfernen
            if len(lemma) <= 2:
                continue

            # Zusätzlicher Stopword-Check
            if lemma in STOP_WORDS:
                continue

            tokens.append(lemma)

        cleaned.append(" ".join(tokens))

    return cleaned


def print_top_words(model, feature_names, n_top_words: int = 10):
    """Gibt pro Topic die Top-Wörter aus (für sklearn LDA/NMF)."""
    for topic_idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[:-n_top_words - 1:-1]
        words = [feature_names[i] for i in top_idx]
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join(words))


def compute_coherence_values(dictionary, corpus, texts, start=2, limit=10, passes=5):
    """
    Berechnet Coherence (c_v) für verschiedene Topic-Anzahlen (k).
    Nutzt gensim LDA + CoherenceModel.
    """
    coherence_values = []

    for num_topics in range(start, limit):
        # gensim LDA für Coherence-Bewertung
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=SEED,
            passes=passes
        )

        # processes=1 verhindert Multiprocessing-Probleme auf macOS/Python 3.12
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1
        )

        coherence_values.append(coherence_model.get_coherence())

    return coherence_values

# Hauptprogramm

def main():
    print("START")

    # 1) Daten laden
    df = pd.read_csv("customercomplaints.csv", low_memory=False)
    texts_all = df["Consumer complaint narrative"].dropna().astype(str)
    print("Anzahl Texte gesamt:", len(texts_all))

    # 2) spaCy-Modell laden (Tokenisierung + Lemmatisierung)
    try:
        # parser/ner deaktiviert => schneller
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' ist nicht installiert.\n"
            "Bitte im aktivierten venv ausführen:\n"
            "  python -m spacy download en_core_web_sm\n"
        )

    # 3) Fixes Subset für Hauptanalyse (Reproduzierbarkeit)
    sample_main = 500
    texts_main = texts_all.sample(n=sample_main, random_state=SEED)
    print("Sample Größe (Hauptanalyse):", len(texts_main))

    # Normalisieren + spaCy-cleaning
    texts_main_norm = [normalize_text(t) for t in texts_main.tolist()]
    cleaned_main = spacy_clean_texts(nlp, texts_main_norm)

    print("\nBeispiel nachher (cleaned):")
    print(cleaned_main[0][:250])

    # 4) Vektorisierung (2 Techniken)
    # Bag-of-Words / Count
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    count_matrix = count_vectorizer.fit_transform(cleaned_main)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_main)

    print("\nVektorisierung abgeschlossen.")
    print("Count-Matrix Shape:", count_matrix.shape)
    print("TFIDF-Matrix Shape:", tfidf_matrix.shape)

    # 5) Topic Modeling (2 Methoden)
    n_topics_initial = 5

    lda = LatentDirichletAllocation(n_components=n_topics_initial, random_state=SEED)
    lda.fit(count_matrix)

    # init="nndsvd" verbessert Stabilität/Interpretierbarkeit (reproduzierbar)
    nmf = NMF(n_components=n_topics_initial, random_state=SEED, init="nndsvd")
    nmf.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={n_topics_initial}) ---")
    print_top_words(lda, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={n_topics_initial}) ---")
    print_top_words(nmf, tfidf_vectorizer.get_feature_names_out())

    # 6) Coherence Score (k-Optimierung) auf fixer Teilmenge
    sample_coh = 300
    texts_coh = texts_all.sample(n=sample_coh, random_state=SEED)
    print("\nBerechne Coherence Scores (Subset)...")
    print("Sample Größe (Coherence):", len(texts_coh))

    texts_coh_norm = [normalize_text(t) for t in texts_coh.tolist()]
    cleaned_coh = spacy_clean_texts(nlp, texts_coh_norm)

    # gensim erwartet tokenisierte Texte
    tokenized_coh = [t.split() for t in cleaned_coh]

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

    # 7) Finale Modelle mit optimalem k trainieren (auf Hauptsample)
    print("\nTrainiere finale Modelle mit optimalem k...")

    lda_final = LatentDirichletAllocation(n_components=best_k, random_state=SEED)
    lda_final.fit(count_matrix)

    nmf_final = NMF(n_components=best_k, random_state=SEED, init="nndsvd")
    nmf_final.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={best_k}) ---")
    print_top_words(lda_final, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={best_k}) ---")
    print_top_words(nmf_final, tfidf_vectorizer.get_feature_names_out())

    print("ENDE")


if __name__ == "__main__":
    main()
