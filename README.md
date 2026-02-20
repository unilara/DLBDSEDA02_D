# NLP Topic Modeling – Consumer Complaint Database

Dieses Projekt analysiert Verbraucherbeschwerden aus der Consumer Complaint Database mithilfe von NLP-Techniken und Topic Modeling.

## Methoden
- Textvorverarbeitung mit spaCy (Tokenisierung, Stopword-Entfernung, Lemmatisierung)
- Bag of Words (CountVectorizer) & TF-IDF
- Topic Modeling mit LDA & NMF
- Bestimmung der optimalen Topic-Anzahl mittels Coherence Score (c_v)

## Datensatz
Der Datensatz ist öffentlich verfügbar auf Kaggle:
https://www.kaggle.com/datasets/selener/consumer-complaintdatabase

## Datenbasis

Für die Analyse wurde ein reproduzierbares Subset von 500 Dokumenten verwendet. Das Subset wurde mit einem festen Random Seed (SEED=42) erzeugt, sodass die Ergebnisse bei jeder Ausführung identisch reproduziert werden können.

## Optimale Topic-Anzahl

Die optimale Anzahl der Topics wurde mittels Coherence Score bestimmt.


## Reproduzierbarkeit
Das Projekt ist vollständig reproduzierbar durch:
- festen Random Seed (SEED=42)
- feste Subset-Größe
- deterministische Initialisierung der Modelle

## Installation
# Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate   (für Mac/Linux) 
oder
venv\Scripts\activate      (für Windows)

# Abhängigkeiten installieren
pip install -r requirements.txt

# spaCy Sprachmodell installieren
python -m spacy download en_core_web_sm

## Ausführung
python analysis.py
