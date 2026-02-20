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


def drop_empty(texts):
    """Entfernt leere/whitespace-only Texte nach Cleaning."""
    return [t for t in texts if isinstance(t, str) and t.strip()]


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

    # n_process=1 => stabil auf allen Systemen (keine Multiprocessing-Probleme)
    for doc in nlp.pipe(texts, batch_size=200, n_process=1):
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
        model = gensim.models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=SEED,
            passes=passes
        )

        # processes=1 verhindert Multiprocessing-Probleme auf macOS/Python 3.12+
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
    if "Consumer complaint narrative" not in df.columns:
        raise SystemExit("Spalte 'Consumer complaint narrative' nicht gefunden. Bitte CSV prüfen.")

    texts_all = df["Consumer complaint narrative"].dropna().astype(str)
    print("Anzahl Texte gesamt:", len(texts_all))

    if len(texts_all) < 10:
        raise SystemExit("Zu wenige Texte nach dropna(). Bitte Datensatz prüfen.")

    # 2) spaCy-Modell laden (Tokenisierung + Lemmatisierung)
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' ist nicht installiert.\n"
            "Bitte im aktivierten venv ausführen:\n"
            "  python -m spacy download en_core_web_sm\n"
        )

    # 3) Fixes Subset für Hauptanalyse (Reproduzierbarkeit)
    sample_main = min(500, len(texts_all))
    texts_main = texts_all.sample(n=sample_main, random_state=SEED)
    print("Sample Größe (Hauptanalyse):", len(texts_main))

    # Normalisieren + spaCy-cleaning
    texts_main_norm = [normalize_text(t) for t in texts_main.tolist()]
    cleaned_main = spacy_clean_texts(nlp, texts_main_norm)
    cleaned_main = drop_empty(cleaned_main)

    if len(cleaned_main) < 10:
        raise SystemExit("Zu wenige Texte nach Cleaning. Bitte Cleaning-Parameter prüfen.")

    print("\nBeispiel nachher (cleaned):")
    print(cleaned_main[0][:250])

    # 4) Vektorisierung (2 Techniken)
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    count_matrix = count_vectorizer.fit_transform(cleaned_main)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_main)

    print("\nVektorisierung abgeschlossen.")
    print("Count-Matrix Shape:", count_matrix.shape)
    print("TFIDF-Matrix Shape:", tfidf_matrix.shape)

    # 5) Topic Modeling (2 Methoden)
    n_topics_initial = 5

    lda = LatentDirichletAllocation(
        n_components=n_topics_initial,
        random_state=SEED,
        learning_method="batch",
        max_iter=20
    )
    lda.fit(count_matrix)

    nmf = NMF(
        n_components=n_topics_initial,
        random_state=SEED,
        init="nndsvd",
        max_iter=400
    )
    nmf.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={n_topics_initial}) ---")
    print_top_words(lda, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={n_topics_initial}) ---")
    print_top_words(nmf, tfidf_vectorizer.get_feature_names_out())

    # 6) Coherence Score (k-Optimierung) auf fixer Teilmenge
    sample_coh = min(300, len(texts_all))
    texts_coh = texts_all.sample(n=sample_coh, random_state=SEED)
    print("\nBerechne Coherence Scores (Subset)...")
    print("Sample Größe (Coherence):", len(texts_coh))
    print("Hinweis: Coherence (c_v) wird via gensim-LDA berechnet und dient als Orientierung für k.")

    texts_coh_norm = [normalize_text(t) for t in texts_coh.tolist()]
    cleaned_coh = spacy_clean_texts(nlp, texts_coh_norm)
    cleaned_coh = drop_empty(cleaned_coh)

    tokenized_coh = [t.split() for t in cleaned_coh]
    tokenized_coh = [t for t in tokenized_coh if len(t) > 0]

    if len(tokenized_coh) < 10:
        raise SystemExit("Zu wenige tokenisierte Texte für Coherence. Bitte Cleaning prüfen.")

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

    lda_final = LatentDirichletAllocation(
        n_components=best_k,
        random_state=SEED,
        learning_method="batch",
        max_iter=20
    )
    lda_final.fit(count_matrix)

    nmf_final = NMF(
        n_components=best_k,
        random_state=SEED,
        init="nndsvd",
        max_iter=400
    )
    nmf_final.fit(tfidf_matrix)

    print(f"\n--- LDA Topics (k={best_k}) ---")
    print_top_words(lda_final, count_vectorizer.get_feature_names_out())

    print(f"\n--- NMF Topics (k={best_k}) ---")
    print_top_words(nmf_final, tfidf_vectorizer.get_feature_names_out())

    print("ENDE")


if __name__ == "__main__":
    main()