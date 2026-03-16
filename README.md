# ML Models Playground

Questo repository contiene una raccolta di modelli di machine learning ed esperimenti con reti neurali sviluppati durante il mio percorso ITS Data Analyst.

L'obiettivo di questo repository è didattico: esplorare diversi approcci di machine learning, comprendere l'intero workflow di un progetto di data science e confrontare algoritmi di machine learning classici con reti neurali.

I progetti inclusi qui si concentrano sulle principali fasi di una pipeline di machine learning:

- esplorazione e visualizzazione dei dati
- preprocessing dei dati e feature engineering
- addestramento di diversi modelli
- valutazione e confronto delle prestazioni dei modelli
- sperimentazione con reti neurali

---

## Struttura della repository
```text
ml-models-playground
│
├── cifar10_cnn
│ ├── cifar10_cnn.ipynb
│ └── cifar10_cnn.py
│
├── tennis_match_prediction
│ ├── tennis_match_prediction.ipynb
│ |── tennis_match_prediction.py
| └── README.md
│
└── README.md
```text

Ogni progetto è disponibile sia come **Jupyter Notebook** (per l'esplorazione e la spiegazione del processo) sia come **script Python**.

---

## Progetti

### Classificazione immagini CIFAR-10

Implementazione di una **Convolutional Neural Network (CNN)** utilizzando **TensorFlow / Keras** per classificare immagini del dataset CIFAR-10.

Elementi principali del progetto:

- preprocessing delle immagini
- progettazione dell'architettura CNN
- addestramento e valutazione del modello
- visualizzazione delle prestazioni durante il training

L'obiettivo di questo esperimento è comprendere come funzionano le reti neurali convoluzionali nei problemi di classificazione di immagini.

---

### Predizione risultati partite di tennis

Pipeline di machine learning progettata per prevedere il risultato di partite di tennis utilizzando statistiche delle partite.

Il progetto include:

- Exploratory Data Analysis (EDA)
- selezione delle feature e pulizia dei dati
- gestione dei valori mancanti
- pipeline di preprocessing con Scikit-learn
- confronto tra diversi modelli di machine learning

Modelli implementati:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Neural Network (Keras)

Il progetto include anche la valutazione dei modelli utilizzando:

- Accuracy
- ROC-AUC
- curve Precision–Recall
- matrici di confusione

---

## Tecnologie utilizzate

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Obiettivo della repository

Questo repository fa parte del mio percorso di apprendimento come **studentessa Data Analyst**.

Serve come spazio per:

- sperimentare con modelli di machine learning
- comprendere il comportamento di diversi algoritmi
- esercitarsi nella costruzione di pipeline complete di machine learning
- documentare il mio percorso di apprendimento attraverso progetti pratici