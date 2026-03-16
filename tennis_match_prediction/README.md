# Tennis Match Prediction

Questo progetto esplora l'uso di tecniche di **machine learning** per prevedere il risultato di partite di tennis utilizzando statistiche di gioco.

L'obiettivo principale del progetto è **didattico**: comprendere l'intero processo di costruzione di un modello di machine learning, dalla preparazione dei dati fino alla valutazione dei modelli.

Per questo motivo sono stati implementati **diversi modelli di machine learning**, anche quando non strettamente necessari, con lo scopo di confrontare approcci diversi e comprenderne il comportamento.

---

# Obiettivo del progetto

Il progetto si propone di:

- analizzare un dataset di statistiche delle partite
- costruire una pipeline di preprocessing robusta
- addestrare diversi modelli di machine learning
- confrontare le prestazioni dei modelli
- sperimentare anche una **rete neurale semplice** a scopo didattico

L'idea non è trovare il modello perfetto, ma **capire il processo di modellazione e confronto tra algoritmi**.

---

# Struttura del progetto

Il progetto è organizzato nelle seguenti fasi principali:

1. Caricamento e preparazione iniziale dei dati  
2. Analisi esplorativa dei dati (EDA)  
3. Decisioni di preprocessing  
4. Creazione della pipeline di preprocessing  
5. Addestramento di modelli classici  
6. Addestramento di una rete neurale  
7. Confronto finale tra modelli

---

# 1. Caricamento del dataset

Il dataset contiene statistiche relative alle partite di tennis, come:

- ace
- doppi falli
- winners
- errori non forzati
- punti a rete
- break points

La variabile target del modello è:

**Result**

che indica quale giocatore ha vinto la partita.

---

# 2. Analisi esplorativa dei dati (EDA)

Prima della fase di modellazione è stata effettuata un'analisi esplorativa dei dati per:

- comprendere la struttura del dataset
- individuare valori mancanti
- analizzare la distribuzione delle feature
- osservare le differenze tra le due classi del target

Durante questa fase sono stati prodotti diversi grafici, tra cui:

- boxplot delle feature per classe
- analisi delle correlazioni
- distribuzioni delle variabili

Questa analisi ha permesso di identificare:

- feature informative
- possibili problemi di data leakage
- strategie di gestione dei valori mancanti

---

# 3. Pulizia dei dati e rimozione del data leakage

Durante l'EDA sono state prese alcune decisioni importanti.

### Rimozione delle colonne categoriche

Alcune colonne contenenti informazioni testuali (come nomi dei giocatori o torneo) sono state rimosse per evitare che il modello imparasse **informazioni legate all'identità dei giocatori**, invece delle statistiche della partita.

### Rimozione di colonne con data leakage

Sono state eliminate alcune variabili che contenevano informazioni troppo vicine al risultato finale della partita, come:

- punteggi dei set
- punteggio finale
- total points won

Queste variabili avrebbero reso il problema artificiale perché il modello avrebbe avuto accesso a informazioni quasi equivalenti al risultato.

---

# 4. Gestione dei valori mancanti

Sono state adottate diverse strategie di imputazione:

- **media** per alcune statistiche di gioco
- **zero** per variabili come ace o doppi falli quando il valore mancava

Questa logica è stata implementata tramite **ColumnTransformer di Scikit-learn**.

---

# 5. Pipeline di preprocessing

Per garantire un flusso corretto di training e valutazione è stata costruita una pipeline di preprocessing che include:

- imputazione dei valori mancanti
- standardizzazione delle feature (per modelli lineari e rete neurale)

Il preprocessing viene **fit solo sul training set** per evitare leakage.

---

# 6. Modelli di Machine Learning

Per scopo didattico sono stati addestrati diversi modelli:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Ogni modello è stato valutato tramite:

- accuracy
- ROC-AUC
- confusion matrix
- curve ROC
- precision-recall curve

---

# 7. Rete neurale

È stata implementata anche una **rete neurale semplice utilizzando Keras**.

L'obiettivo non era necessariamente ottenere prestazioni migliori rispetto ai modelli classici, ma comprendere:

- come costruire una rete neurale
- come gestire il training con validation
- come utilizzare tecniche come early stopping

La rete neurale è stata addestrata utilizzando gli stessi dati preprocessati dei modelli classici.

---

# 8. Confronto finale dei modelli

Infine, tutti i modelli sono stati confrontati utilizzando diverse metriche:

- Accuracy
- ROC-AUC
- Average Precision

Sono stati inoltre generati grafici comparativi come:

- ROC Curve
- Precision–Recall Curve

Questo confronto permette di osservare come modelli diversi si comportano sullo stesso problema.

---

# Tecnologie utilizzate

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Nota sul progetto

Questo progetto è stato sviluppato principalmente con **finalità didattiche**.

Per questo motivo sono stati implementati diversi modelli e analisi anche quando non strettamente necessari, con l'obiettivo di:

- comprendere meglio il comportamento degli algoritmi
- esercitarsi nella costruzione di pipeline complete di machine learning
- confrontare approcci diversi allo stesso problema.