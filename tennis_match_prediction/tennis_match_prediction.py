

"""# **FASE 1   -  CARICAMENTO DATI**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score
)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# A) CARICAMENTO DATASET
# ----------------------------
tennis = fetch_ucirepo(id=300)

X_raw = tennis.data.features
y_raw = tennis.data.targets

df_raw = pd.concat([X_raw, y_raw], axis=1)

# Controlli base
print("Shape df:", df_raw.shape)
print("\nInfo:")
print(df_raw.info())
df_raw.head()

"""# **FASE 2  - Ramo EDA su copia (analisi esplorativa)**"""

# ----------------------------
#  CREO COPIA PER EDA
# ----------------------------

df_eda = df_raw.copy()

# ----------------------------
#  SEPARO X e y
# ----------------------------

# Target
y_eda = df_eda["Result"]

# Features
X_eda = df_eda.drop(columns=["Result"])

# ----------------------------
#  Controllo missing values
# ----------------------------

# Missing values (top 20)
missing = X_eda.isnull().sum().sort_values(ascending=False)
print("\nTotale null: ", X_eda.isnull().sum().sum())
print("\nTop 20 colonne con più NaN:")
print(missing.head(20))

# ----------------------------
#  Colonne categoriche (stringhe)
# ----------------------------

cat_cols = X_eda.select_dtypes(include=["object", "string"]).columns
num_cols = X_eda.select_dtypes(include=np.number).columns

print("Colonne categoriche:", list(cat_cols))
print("\nNumero colonne numeriche:", len(num_cols))
print("\nVecchia shape:", X_eda.shape)

# decisione didattica elimino le colonne categoriche
    # elimino Player1 / Player2 perche cosi evito il bias da identita.
    # Voglio che il modello impara statistiche e non che Federer = vittoria
    # elimino Tournament perche voglio modello basato su statistiche match
    # e non su dove si è svolto. È una semplificazione didattica.

X_eda = X_eda.drop(cat_cols, axis=1)
print("\nNuova shape:", X_eda.shape)

"""   Elimino le colonne

Colonna	Significato
ST1	    Set 1
ST2	    Set 2
ST3	    Set 3
ST4	    Set 4
ST5	    Set 5

--> Sono statistiche dei set  quindi  Null ≠ dato mancante ma Null = set non giocato
Decisione didattica: Eliminiamo tutte le colonne dei set perche troppi null e informazione già indiretta nel risultato

TPW (Total Points Won)

--> Totale punti vinti. È quasi il risultato della partita. Data leakage forte!
Eliminiamo anche TPW.

FNL1 / FNL2    Final score

--> data leakage. Contiene risultato match. Eliminiamo.
"""

# Definisco le colonne da eliminare (Data Leakage + Target)
colonne_da_eliminare = [
    'ST1.1', 'ST1.2',         # Game set 1
    'ST2.1', 'ST2.2',         # Game set 2
    'ST3.1', 'ST3.2',         # Game set 3
    'ST4.1', 'ST4.2',         # Game set 4
    'ST5.1', 'ST5.2',         # Game set 5
    'TPW.1', 'TPW.2',         # Punti totali (troppo vicini al risultato)  leakage!!
    'FNL1',  'FNL2'           # Risultato finale set                       leakage!!
]
print("Vecchia shape:", X_eda.shape)

X_eda = X_eda.drop(colonne_da_eliminare, axis=1)

print("Features rimaste in X:")
print(X_eda.columns.tolist())
print("Nuova shape:", X_eda.shape)

# ---------------------------------
#  Controllo missing values rimasti
# ---------------------------------

# Missing values (top 20)
missing = X_eda.isnull().sum().sort_values(ascending=False)
print("\nTotale null: ", X_eda.isnull().sum().sum())
print("\nTop 20 colonne con più NaN:")
print(missing.head(20))

"""   Gestisco i valori null       su copia di dataset per EDA

UFE - Unforced Errors
WNR

--> Winners
Null pochi (126 su dataset grande). Qui ha senso imputare.
Strategia: Media.

NPA / NPW  

--> Net Points Attempted / Won
Null pochi - imputazione media.

ACE / DBF

--> Ace e Doppi falli.
Null pochissimi - logico usare 0. Se non registrato - probabilmente 0

BPC	- Break Points Created

BPW	- Break Points Won

-->
Strategia: Media imputazione perché dataset grande, pochi null e variabile continua

"""

# Media imputazione
for col in ["UFE.1","UFE.2","WNR.1","WNR.2",
            "NPA.1","NPA.2","NPW.1","NPW.2",
            "BPC.1","BPW.1","BPC.2","BPW.2"]:
    X_eda[col] = X_eda[col].fillna(X_eda[col].mean())

# Zero imputazione
for col in ["ACE.1","ACE.2","DBF.1","DBF.2"]:
        X_eda[col] = X_eda[col].fillna(0)

print("\nTotale null: ", X_eda.isnull().sum().sum())

"""#   Grafici EDA

Boxplot per classe (Result=0 vs Result=1)
                    
Serve per verificare se:

La mediana (linea nel box) è diversa tra 0 e 1?

I box sono separati o sovrapposti?

 Una classe ha valori molto più alti?
    
Se sì → quella feature distingue bene le classi e quindi aiuta a prevedere Result
"""

# unisco per i boxplot

df_plot = X_eda.copy()
df_plot["Result"] = y_eda # Serve perché Seaborn ha bisogno di avere X e y nello stesso dataframe
                          # per fare grafici "per classe"

cols = [c for c in df_plot.columns if c != "Result"] # lista con tutti i nomi delle feature,
                                                     # escludo la colonna target.

per_fig = 5   # Numero di boxplot che voglio in ogni figura.
n_fig = math.ceil(len(cols) / per_fig) # Calcolo quante figure servono in totale.
                                       # Uso ceil per arrotondare sempre verso l’alto.

for f in range(n_fig):
    subset = cols[f*per_fig:(f+1)*per_fig] # Seleziono solo le feature che andranno in questa figura

    fig, axes = plt.subplots(1, 5, figsize=(16, 5))  # creo 5 spazi vuoti dove puoi disegno grafici.
    axes = axes.ravel()   # Trasformo la matrice di assi in una lista semplice
                          # Serve per iterare facilmente con zip.

    for ax, col in zip(axes, subset): # zip associa: ogni asse grafico alla sua feature da plottare
        sns.boxplot(data=df_plot, x="Result", y=col, ax=ax)  # ax=ax decide dove disegnare il grafico
                                  # Result ha due valori: 0 e 1
                                  # Seaborn automaticamente:
                                  # prende tutte le righe con Result = 0
                                  # prende tutte le righe con Result = 1
                                  # costruisce un box per ogni gruppo
        ax.set_title(col)
        ax.set_xlabel("Result")
        ax.set_ylabel("")   # Tolgo etichetta Y per non appesantire il layout

    for ax in axes[len(subset):]:
        ax.axis("off")  # Se l’ultima figura ha meno di 5 feature,
                        # alcuni subplot restano vuoti.
                        # Qui li nascondo per pulizia visiva.

    plt.tight_layout()  # Sistema automaticamente spazi e margini
    plt.show()

plt.figure(figsize=(20,7))
df_plot.boxplot()
plt.xticks(rotation=90)
plt.show()

# ----------------------------
# correlazioni featurs e target
# ----------------------------

df_numeric = df_plot.select_dtypes(include=np.number)

corr_target = df_numeric.corr()['Result'].sort_values(ascending=False)

                      # .corr() - correlation matrix, calcola quanto le variabili sono legate tra loro.
                      # .corr()['Result'] - correlazione di ogni feature con il target

print("\nCorrelazione (imputazione globale SOLO EDA) ", corr_target)

"""

# **FASE 3 — Decisioni di preprocessing**
##  RAMO TRAINING

aplico al dataset di training cambiamenti che ritengo opportune dopo valutazione su dataset di EDA

Qui faccio:
1) decido cosa togliere (categoriche + leakage)
2) split train/val/test
3) preprocessing con ColumnTransformer (mean + zero) + StandardScaler, fit solo su train
4) modelli classici + rete neurale"""

# ----------------------------
#  CREO COPIA PER TRAINING
# ----------------------------

#  Parto dal RAW (non dalla copia imputata EDA)
df = df_raw.copy()

y = df["Result"]
X = df.drop(columns=["Result"])

# ----------------------------
#  RIMOZIONE COLONNE CATEGORICHE
# ----------------------------

# 1) Drop categoriche (scelta didattica)
cat_cols = X.select_dtypes(include=["object", "string"]).columns
X = X.drop(columns=cat_cols)

# 2) Drop leakage (se colonne presenti)
leak_cols = [
    'ST1.1', 'ST1.2',
    'ST2.1', 'ST2.2',
    'ST3.1', 'ST3.2',
    'ST4.1', 'ST4.2',
    'ST5.1', 'ST5.2',
    'TPW.1', 'TPW.2',
    'FNL1',  'FNL2'
]
leak_present = [c for c in leak_cols if c in X.columns]  # un controllo in piu, verifico se le colonne sono presenti
X = X.drop(columns=leak_present)

print("Shape X dopo drop:", X.shape)

"""# **FASE 4 — Split**"""

# ----------------------------
#  SPLIT train/val/test
# ----------------------------

# 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 20% di (train+val) diventa val -> 0.15 del temp = 0.2 totale
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15,
    random_state=42,
    stratify=y_temp
)

print("\nTrain:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

"""# **FASE 5 — Preprocessing statistico**  (imputazione mista + scaling)"""

# Controllo missing values

# print(X.isnull().sum().sort_values(ascending=False))
print("\nTotale null: ", X.isnull().sum().sum())

# ----------------------------
#  PREPROCESSING: imputazione + scaling
# ----------------------------

zero_cols = ["ACE.1","ACE.2","DBF.1","DBF.2"]
mean_cols = [col for col in X.columns if col not in zero_cols]

# versione 1
preprocessor = ColumnTransformer(
    transformers=[
        ("mean_imp", SimpleImputer(strategy="mean"), mean_cols),
        ("zero_imp", SimpleImputer(strategy="constant", fill_value=0), zero_cols)
    ])

# oppure 
preprocessor = ColumnTransformer([
        ("mean_imp", SimpleImputer(strategy="mean"), mean_cols),
        ("zero_imp", SimpleImputer(strategy="constant", fill_value=0), zero_cols)
    ])


# versione 2
        # mean_imputer = SimpleImputer(strategy="mean")

        # zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         # (nome, trasformazione, colonne)
        #         ("mean_imp", mean_imputer, mean_cols),
        #         ("zero_imp", zero_imputer, zero_cols)
        #     ]
        # )

# versione 3

        # # mini-pipeline per colonne con imputazione media
        # mean_pipeline = Pipeline([
        #     ("imputer", SimpleImputer(strategy="mean"))
        # ])

        # # mini-pipeline per colonne con imputazione zero
        # zero_pipeline = Pipeline([
        #     ("imputer", SimpleImputer(strategy="constant", fill_value=0))
        # ])

        # # unisco le due mini-pipeline nel ColumnTransformer
        # preprocessor = ColumnTransformer([
        #     ("mean_part", mean_pipeline, mean_cols),
        #     ("zero_part", zero_pipeline, zero_cols)
        # ])

# pipeline per classici

# Preprocessing per alberi
pipeline_tree = Pipeline([
    ("imputer", preprocessor)   # senza scaler
])

# Preprocessing per modelli lineari
pipeline_scaled = Pipeline([
    ("imputer", preprocessor),
    ("scaler", StandardScaler())
])


# pipeline per rete neurale

pipeline_scaled_nn = Pipeline([
    ("imputer", preprocessor),
    ("scaler", StandardScaler())
])

# ------------------------------------------------------------
#  DATI PER MODELLI CLASSICI: uso TEMP (train+val)
# ------------------------------------------------------------
X_classic = X_temp
y_classic = y_temp

X_classic_tree   = pipeline_tree.fit_transform(X_classic)
X_test_tree_cv   = pipeline_tree.transform(X_test)        # test trasformato con lo stesso preprocessing

X_classic_scaled = pipeline_scaled.fit_transform(X_classic)
X_test_scaled_cv = pipeline_scaled.transform(X_test)

print("Shape classic_tree:", X_classic_tree.shape, "Shape test_tree:", X_test_tree_cv.shape)
print("Shape classic_scaled:", X_classic_scaled.shape, "Shape test_scaled:", X_test_scaled_cv.shape)


# ------------------------------------------------------------
#  DATI PER RETE NEURALE
# ------------------------------------------------------------
X_train_scaled = pipeline_scaled_nn.fit_transform(X_train)
X_val_scaled   = pipeline_scaled_nn.transform(X_val)
X_test_scaled  = pipeline_scaled_nn.transform(X_test)

print("\nShape train per rete neurale scaled: ", X_train_scaled.shape,
      "\nShape val per rete neurale scaled: ", X_val_scaled.shape,
      "\nShape test per rete neurale scaled: ", X_test_scaled.shape, )

"""# **FASE 6 — Modelli classici**  (sklearn)"""

# ------------------------------------------------------------
# 1) DEFINISCO I NOMI DELLE FEATURE DOPO IL PREPROCESSOR
# ------------------------------------------------------------
# ColumnTransformer produce colonne nell'ordine:
# prima mean_cols, poi zero_cols.
# Questo serve per feature importance / coefs / plot_tree.

feature_names = mean_cols + zero_cols

# ------------------------------------------------------------
# 2) CROSS-VALIDATION (versione semplice con cross_val_score)
# ------------------------------------------------------------
# La CV la facciamo su X_classic (non sul test).
# Usiamo i dati già preprocessati:
# - X_classic_tree per modelli ad albero (tree/rf/gb)
# - X_classic_scaled per LogisticRegression

# cross_val_score(
#     estimator,
#     X,
#     y=None,
#     *,
#     groups=None,
#     scoring=None,
#     cv=None,
#     n_jobs=None,
#     verbose=0,
#     fit_params=None,
#     pre_dispatch="2*n_jobs",
#     error_score=np.nan
# )


# ---------- Decision Tree ----------
tree_model = DecisionTreeClassifier(random_state=42)

tree_auc_scores = cross_val_score(tree_model, X_classic_tree, y_classic, cv=5, scoring="roc_auc")
tree_acc_scores = cross_val_score(tree_model, X_classic_tree, y_classic, cv=5, scoring="accuracy")

tree_cv = (tree_acc_scores.mean(), tree_acc_scores.std(), tree_auc_scores.mean(), tree_auc_scores.std())


# ---------- Random Forest ----------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

rf_auc_scores = cross_val_score(rf_model, X_classic_tree, y_classic, cv=5, scoring="roc_auc")
rf_acc_scores = cross_val_score(rf_model, X_classic_tree, y_classic, cv=5, scoring="accuracy")

rf_cv = (rf_acc_scores.mean(), rf_acc_scores.std(), rf_auc_scores.mean(), rf_auc_scores.std())


# ---------- Gradient Boosting ----------
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)

gb_auc_scores = cross_val_score(gb_model, X_classic_tree, y_classic, cv=5, scoring="roc_auc")
gb_acc_scores = cross_val_score(gb_model, X_classic_tree, y_classic, cv=5, scoring="accuracy")

gb_cv = (gb_acc_scores.mean(), gb_acc_scores.std(), gb_auc_scores.mean(), gb_auc_scores.std())


# ---------- Logistic Regression (usa scaled) ----------
log_model = LogisticRegression(max_iter=2000)

log_auc_scores = cross_val_score(log_model, X_classic_scaled, y_classic, cv=5, scoring="roc_auc")
log_acc_scores = cross_val_score(log_model, X_classic_scaled, y_classic, cv=5, scoring="accuracy")

log_cv = (log_acc_scores.mean(), log_acc_scores.std(), log_auc_scores.mean(), log_auc_scores.std())


# ---------- Tabella riassuntiva ----------
# tree_cv[0] → CV_Acc_Mean
# tree_cv[1] → CV_Acc_Std
# tree_cv[2] → CV_AUC_Mean
# tree_cv[3] → CV_AUC_Std

rows = []

rows.append(["DecisionTree",       tree_cv[0], tree_cv[1], tree_cv[2], tree_cv[3]])
rows.append(["RandomForest",       rf_cv[0],   rf_cv[1],   rf_cv[2],   rf_cv[3]])
rows.append(["GradientBoosting",   gb_cv[0],   gb_cv[1],   gb_cv[2],   gb_cv[3]])
rows.append(["LogisticRegression", log_cv[0],  log_cv[1],  log_cv[2],  log_cv[3]])

cv_df = pd.DataFrame(
    rows,
    columns=["Model", "CV_Acc_Mean", "CV_Acc_Std", "CV_AUC_Mean", "CV_AUC_Std"]
)

cv_df = cv_df.sort_values(by="CV_AUC_Mean", ascending=False)

print("\nRIASSUNTO CROSS-VALIDATION:")
print(cv_df)

# ------------------------------------------------------------
# 3) FIT FINALE SU TUTTO X_classic + VALUTAZIONE SU TEST
# ------------------------------------------------------------
# Qui torniamo al tuo flusso classico: alleno 1 volta e valuto su test.
# Ma ora alleno su X_classic (più dati).

# Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_classic_tree, y_classic)
tree_proba = tree.predict_proba(X_test_tree_cv)[:, 1]
tree_pred = (tree_proba > 0.5).astype(int)

cm_tree = confusion_matrix(y_test, tree_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_tree)
disp.plot()
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_classic_tree, y_classic)
rf_proba = rf.predict_proba(X_test_tree_cv)[:, 1]
rf_pred = (rf_proba > 0.5).astype(int)

cm_rf = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
gb.fit(X_classic_tree, y_classic)
gb_proba = gb.predict_proba(X_test_tree_cv)[:, 1]
gb_pred = (gb_proba > 0.5).astype(int)

cm_gb = confusion_matrix(y_test, gb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_gb)
disp.plot()
plt.title("Gradient Boosting - Confusion Matrix")
plt.show()

# Logistic Regression
log = LogisticRegression(max_iter=2000)
log.fit(X_classic_scaled, y_classic)
log_proba = log.predict_proba(X_test_scaled_cv)[:, 1]
log_pred = (log_proba > 0.5).astype(int)

cm_log = confusion_matrix(y_test, log_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# ------------------------------------------------------------
# 4) TABELLA results
# ------------------------------------------------------------
def metrics_binary(name, y_real, y_pred, y_proba):
    acc = accuracy_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_proba)
    return {"model": name, "accuracy": acc, "auc": auc}

results = []
results.append(metrics_binary("Decision Tree", y_test, tree_pred, tree_proba))
results.append(metrics_binary("Random Forest", y_test, rf_pred, rf_proba))
results.append(metrics_binary("Gradient Boosting", y_test, gb_pred, gb_proba))
results.append(metrics_binary("Logistic Regression", y_test, log_pred, log_proba))

pd.DataFrame(results).sort_values(by="auc", ascending=False)

"""# Feature Importance - Random Forest"""

#    Estrazione Feature Importance

feature_importance_df = pd.DataFrame({
    # nome_colonna: valori
    "Feature": feature_names,   # mean_cols + zero_cols
    "Importance": rf.feature_importances_  # è un attributo del modello Random Forest dopo il fit.
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Grafico Feature Importance

plt.figure(figsize=(10, 8))
plt.barh(
    feature_importance_df["Feature"],  # y → etichette sull’asse verticale,
    feature_importance_df["Importance"]  # width → lunghezza barra
)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importanza")
plt.ylabel("Feature")
plt.show()

"""# Feature Importance — Logistic Regression"""

from matplotlib.patches import Patch

coefs = pd.DataFrame({
    "Feature": feature_names,   # mean_cols + zero_cols
    "Coefficient": log.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

colors = ['firebrick' if x > 0 else 'steelblue'
          for x in coefs['Coefficient']]

plt.figure(figsize=(10, 6))
plt.barh(coefs['Feature'], coefs['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title("Feature Importance (Coefficienti della Regressione)")
plt.xlabel("Impatto sulla probabilità di vittoria")
legend_elements = [
    Patch(facecolor='firebrick', label='Favorisce Player 1'),
    Patch(facecolor='steelblue', label='Favorisce Player 2')
]
plt.legend(handles=legend_elements)
plt.show()

"""# **FASE 7 — Rete neurale (Keras)**
Usa gli stessi dati preprocessati (X_train_slaled, X_val_scaled, X_test_scaled).

Keras vuole numpy array
"""

# Keras lavora con numpy array ma y_train è una pandas Series --> trasformo

y_train_np = np.array(y_train)
y_val_np   = np.array(y_val)
y_test_np  = np.array(y_test)

n_features = X_train_scaled.shape[1] # num colonne

nn = Sequential([
    Input(shape=(n_features,)),
    Dense(8, activation="relu"),
    # Dense(2, activation="relu"),
    Dense(1, activation="sigmoid")
])

nn.compile(
    optimizer=SGD(learning_rate=0.05),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="best_tennis_nn.keras",
    monitor="val_loss",
    save_best_only=True
)

# loss           → errore su train
# accuracy       → accuracy su train
# val_loss       → errore su validation
# val_accuracy   → accuracy su validation

history = nn.fit(
    X_train_scaled, y_train_np,
    validation_data=(X_val_scaled, y_val_np),
    epochs=100,
    batch_size=15,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
best_nn = load_model("best_tennis_nn.keras")

test_loss, test_acc = best_nn.evaluate(X_test_scaled, y_test_np, verbose=0)
print("NN test_loss:", test_loss)
print("NN test_acc :", test_acc)

nn_proba = best_nn.predict(X_test_scaled, verbose=0).reshape(-1)   # un vettore 1D di probabilità
nn_pred = (nn_proba > 0.5).astype(int)

nn_auc = roc_auc_score(y_test_np, nn_proba) # AUC misura quanto bene il modello separa le classi per tutte le soglie possibili
print("NN AUC:", nn_auc)

# Aggiungo NN alla tabella risultati
results.append(metrics_binary("NeuralNetwork(Keras)", y_test_np, nn_pred, nn_proba))
pd.DataFrame(results).sort_values(by="auc", ascending=False)

"""##  Confronto ROC curve (tutti i modelli)"""

plt.figure(figsize=(8,6))

for name, proba in [
    ("Decision Tree", tree_proba),
    ("Random Forest", rf_proba),
    ("Gradient Boosting", gb_proba),
    ("Logistic Regression", log_proba),
    ("Neural Network (Keras)", nn_proba),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Confronto Modelli")
plt.legend()
plt.show()

"""##  Confronto Precision–Recall curve (tutti i modelli)"""

plt.figure(figsize=(8, 6))

for name, proba in [
    ("Decision Tree", tree_proba),
    ("Random Forest", rf_proba),
    ("Gradient Boosting", gb_proba),
    ("Logistic Regression", log_proba),
    ("Neural Network (Keras)", nn_proba),
]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)  # AP standard
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Confronto Modelli")
plt.legend()
plt.show()

"""## Grafici training NN (loss/accuracy)"""

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("NN Loss durante training")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("NN Accuracy durante training")
plt.xlabel("Epoche")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

"""# VISUALIZZAZIONE ALBERO DECISIONALE"""

feature_names = mean_cols + zero_cols

plt.figure(figsize=(20, 10))
plot_tree(tree,
          feature_names=feature_names,
          class_names=['Vince P2', 'Vince P1'],
          filled=True,
          rounded=True,
          fontsize=7)
plt.title("Logica del Decision Tree")
plt.show()

"""# Tabella Confronto Finale"""

# ----------------------------
# Calcolo metriche
# ----------------------------
results = []

for name, pred, proba in [
    ("Decision Tree", tree_pred, tree_proba),
    ("Random Forest", rf_pred, rf_proba),
    ("Gradient Boosting", gb_pred, gb_proba),
    ("Logistic Regression", log_pred, log_proba),
    ("Neural Network (Keras)", nn_pred, nn_proba),
]:

    acc = accuracy_score(y_test, pred)
    auc_roc = roc_auc_score(y_test, proba)
    ap_pr = average_precision_score(y_test, proba)

    results.append([name, acc, auc_roc, ap_pr])


# ----------------------------
# Creo DataFrame
# ----------------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "AUC_ROC", "AP_PR"]
)

# Ordino per AUC_ROC (facoltativo)
results_df = results_df.sort_values(by="AUC_ROC", ascending=False)

print("\n Confronto Finale Modelli \n")
print(results_df)