# CIFAR-10 Image Classification with Convolutional Neural Network

Questo progetto esplora l'utilizzo di una **rete neurale convoluzionale (CNN)** per la classificazione di immagini del dataset **CIFAR-10**.

L'obiettivo principale del progetto è **didattico**: comprendere come funzionano le reti neurali convoluzionali e come costruire un semplice sistema di classificazione di immagini utilizzando **TensorFlow e Keras**.

---

# Obiettivo del progetto

Questo progetto è stato sviluppato per:

- comprendere il funzionamento delle **Convolutional Neural Networks**
- imparare a costruire un modello di deep learning con **Keras**
- sperimentare il processo di **training di una rete neurale**
- osservare il comportamento del modello durante l'addestramento
- analizzare le prestazioni del modello sul dataset di test

---

# Il dataset CIFAR-10

Il dataset **CIFAR-10** è un dataset molto utilizzato per esercizi di computer vision e deep learning.

Contiene:

- **60.000 immagini a colori**
- dimensione **32 × 32 pixel**
- **10 classi di oggetti**

Le classi presenti nel dataset sono:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Il dataset è diviso in:

- **training set**
- **test set**

---

# Struttura del modello

Il modello utilizzato in questo progetto è una **Convolutional Neural Network (CNN)** costruita con **Keras**.

Una CNN è particolarmente adatta alla classificazione di immagini perché riesce a:

- individuare pattern visivi
- riconoscere forme e strutture nelle immagini
- ridurre la dimensionalità mantenendo le informazioni più rilevanti

La rete neurale include tipicamente:

- layer convoluzionali
- funzioni di attivazione
- layer di pooling
- layer completamente connessi (dense)

---

# Pipeline del progetto

Il progetto segue le principali fasi di un workflow di deep learning.

## 1. Caricamento del dataset

Il dataset CIFAR-10 viene caricato tramite le utility fornite da **TensorFlow / Keras**.

Le immagini vengono separate in:

- training set
- test set

---

## 2. Preprocessing delle immagini

Prima dell'addestramento del modello, le immagini vengono preprocessate.

In questa fase vengono eseguite operazioni come:

- normalizzazione dei pixel
- conversione dei dati nel formato richiesto dalla rete neurale

Questo passaggio è importante perché migliora la stabilità del training.

---

## 3. Costruzione della rete neurale

Il modello CNN viene costruito utilizzando **Keras Sequential API**.

La rete include diversi layer che permettono di:

- estrarre caratteristiche dalle immagini
- ridurre progressivamente la dimensionalità
- classificare l'immagine in una delle 10 classi

---

## 4. Addestramento del modello

Durante il training il modello impara a riconoscere le classi delle immagini analizzando il training set.

Il processo di training include:

- funzione di perdita (loss function)
- ottimizzatore
- aggiornamento iterativo dei pesi della rete

Durante l'addestramento vengono monitorate metriche come:

- loss
- accuracy

---

## 5. Valutazione del modello

Una volta completato il training, il modello viene valutato sul **test set**, che contiene dati non utilizzati durante l'addestramento.

Questo permette di misurare la capacità del modello di **generalizzare su nuove immagini**.

---

# Tecnologie utilizzate

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

# Scopo didattico del progetto

Questo progetto è stato sviluppato con **finalità didattiche**, per comprendere:

- i principi base del deep learning
- la struttura delle convolutional neural networks
- il processo di training e valutazione di un modello di classificazione di immagini

Il focus principale è quindi **l'apprendimento del processo e degli strumenti**, più che l'ottimizzazione estrema delle prestazioni del modello.