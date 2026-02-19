# NLP Topic Modeling – Consumer Complaint Database

Dieses Projekt analysiert Verbraucherbeschwerden aus der Consumer Complaint Database 
mithilfe von NLP-Techniken.

## Methoden
- Textvorverarbeitung mit spaCy
- CountVectorizer & TF-IDF
- Topic Modeling mit LDA und NMF
- Bestimmung der optimalen Topic-Anzahl mittels Coherence Score

## Datensatz
Der Datensatz ist öffentlich verfügbar auf Kaggle:
https://www.kaggle.com/datasets/selener/consumer-complaintdatabase

## Datenbasis

Für die Analyse wurde ein Subset von 500 Dokumenten verwendet.

## Optimale Topic-Anzahl

Die optimale Anzahl der Topics wurde mittels Coherence Score bestimmt.
Ergebnis: k = 3

## Ausführung
python analysis.py
