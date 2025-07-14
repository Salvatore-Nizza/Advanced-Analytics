Progetto Advanced Analytics | start2impact University
Questo repository contiene il notebook del progetto sviluppato nell'ambito del Corso Advanced Analytics di start2impact University. Il progetto esplora diverse tecniche di analisi dei dati e di machine learning, spaziando dalla regressione alla classificazione, fino all'apprendimento non supervisionato e all'analisi delle serie temporali.
📚 Librerie Principali Utilizzate
Il progetto fa uso delle seguenti librerie Python, essenziali per la manipolazione dei dati, l'analisi statistica, la visualizzazione e la costruzione di modelli di Machine Learning:
• numpy per operazioni numeriche e calcoli statistici.
• pandas per la manipolazione e l'analisi dei dati tabellari.
• scipy (in particolare scipy.stats) per calcoli statistici come la moda e la skewness.
• matplotlib.pyplot per la visualizzazione dei dati.
• google.colab.drive per l'accesso a Google Drive (se il notebook viene eseguito su Google Colab).
• sklearn.preprocessing (LabelEncoder, StandardScaler, PolynomialFeatures) per la pre-elaborazione dei dati e la creazione di feature polinomiali.
• sklearn.model_selection (train_test_split) per la suddivisione del dataset in set di training e test.
• sklearn.linear_model (LinearRegression, LogisticRegression) per l'implementazione dei modelli di regressione lineare e logistica.
• sklearn.metrics (mean_squared_error, mean_absolute_error, r2_score, classification_report, f1_score, confusion_matrix) per la valutazione delle performance dei modelli.
• sklearn.tree (DecisionTreeClassifier) per l'implementazione del modello ad albero decisionale.
• sklearn.cluster (KMeans) per l'implementazione dell'algoritmo di clustering K-Means.
• seaborn per la visualizzazione avanzata (es. heatmap per la confusion matrix).
📊 Esplorazione del Dataset e Pre-elaborazione (Supermarket Sales)
Il primo dataset utilizzato è supermarket_sales.csv, che contiene dati sulle vendite di un supermercato.
1. Caricamento e Ispezione Iniziale: Il dataset viene caricato e le prime 100 occorrenze vengono visualizzate per una comprensione intuitiva dei dati.
2. Metadati e Pulizia:
    ◦ Viene ispezionato il tipo di dato e l'assenza di valori nulli.
    ◦ Colonne ritenute poco utili per l'analisi predittiva, come 'Invoice ID', 'Tax 5%', 'Total', 'Date', 'Time', 'cogs', 'gross margin percentage', vengono rimosse.
📈 Analisi Statistica Descrittiva
Per la colonna Rating, che è la variabile target (label) per la predizione, sono state calcolate le seguenti statistiche:
• Media del Rating: 6.78.
• Mediana del Rating: 6.8.
• Moda del Rating: Modalità(5.9) Frequenza(73).
• Deviazione Standard del Rating: 1.72.
La distribuzione dei rating è risultata essere più o meno uniforme e senza skewness.
In contrasto, la colonna gross income (utile lordo) mostra una significativa skewness, con la maggioranza degli ordini che presentano un basso reddito lordo. Il valore esatto della skewness per gross income è 0.95.
🛠️ Feature Engineering
Per preparare i dati ai modelli di machine learning, sono state applicate le seguenti tecniche:
1. Encoding delle Variabili Categoriche: Colonne come 'Branch', 'City', 'Customer type', 'Gender', 'Product line' e 'Payment' sono state trasformate da categoriche a numeriche utilizzando LabelEncoder. Questo è fondamentale poiché gli algoritmi di machine learning richiedono input numerici.
    ◦ Viene mostrata la mappatura dei valori originali ai valori codificati per ciascuna colonna.
2. Feature Scaling (Standardizzazione): Le colonne 'Unit price' e 'gross income' sono state standardizzate utilizzando StandardScaler. Questa operazione aiuta a migliorare le performance dei modelli, specialmente per 'gross income' che presentava una forte skewness. La standardizzazione viene verificata controllando che la media delle colonne sia prossima a 0 e la deviazione standard a 1.
🧪 Training e Valutazione dei Modelli di Regressione
Il dataset pre-processato viene diviso in set di training (80%) e test (20%) per la valutazione dei modelli di regressione, con Rating come variabile target.
1. Regressione Lineare
• Un modello di LinearRegression è stato addestrato per predire i Rating.
• Metriche di Valutazione:
    ◦ Mean Squared Error (MSE): 2.97.
    ◦ Mean Absolute Error (MAE): 1.35.
    ◦ R2-score: -0.0003.
• Conclusione: L'R2-score estremamente basso indica che il modello lineare è peggiore rispetto a una semplice media dei valori, suggerendo che un modello lineare non cattura adeguatamente la relazione nei dati.
2. Regressione Polinomiale
• È stata tentata una PolynomialRegression per migliorare le performance, inizialmente con un grado del polinomio pari a 2.
• Metriche di Valutazione (Grado 2):
    ◦ MSE: 2.99.
    ◦ MAE: 1.36.
    ◦ R2-score: -0.0080.
• Conclusione (Grado 2): Le performance sono peggiorate rispetto alla regressione lineare semplice.
• Esperimenti con Gradi Maggiori: Sono stati testati gradi del polinomio da 3 a 5.
    ◦ Grado 3: MSE = 3.01, MAE = 1.37, R2 = -0.0093.
    ◦ Grado 4: MSE = 3.03, MAE = 1.37, R2 = -0.0175.
    ◦ Grado 5: MSE = 3.03, MAE = 1.37, R2 = -0.0175.
• Spiegazione del Peggioramento: L'aumento del grado del polinomio non ha portato a miglioramenti, anzi, le performance sono peggiorate. Questo suggerisce che la causa potrebbe essere l'overfitting dei dati, probabilmente dovuto a un dataset dimensionalmente limitato (con pochi dati), il che comporta una ridotta capacità di generalizzazione del modello. Non sono stati riscontrati outlier significativi che possano giustificare il peggioramento delle performance.
🍎 Classificazione della Qualità delle Mele
Per un problema di classificazione, è stato introdotto un nuovo dataset: apple_quality.csv. L'obiettivo è classificare la qualità delle mele.
1. Caricamento e Pulizia: Il dataset viene caricato. Viene rimossa l'ultima riga, che presentava valori nulli.
2. Encoding della Label: La colonna 'Quality', che è la label categorica, viene encodata numericamente utilizzando LabelEncoder ('Good' e 'Bad' vengono mappate).
3. Train/Test Split: Il dataset viene nuovamente diviso in training (80%) e test (20%).
1. Regressione Logistica
• Un modello LogisticRegression è stato addestrato per la classificazione della qualità delle mele.
• Metrica Iniziale (F1-score): L'obiettivo è superare un F1-score di 0.80, massimizzando precision e recall.
• Ottimizzazione del Threshold: È stato cercato il miglior threshold per la probabilità di predizione (predict_proba) per massimizzare l'F1-score.
    ◦ Miglior Threshold Trovato: 0.49.
    ◦ Miglior F1-score (media ponderata): 0.80.
    ◦ Analisi delle Metriche dopo Thresholding: Si è notato che l'F1-score sulla label '1' (qualità buona) migliora a scapito dell'altra label, mentre le medie 'macro avg' e 'weighted avg' per F1-score e recall rimangono invariate. La precision migliora sulla label '2' a discapito dell'altra, e le medie 'macro avg' e 'weighted avg' per la precision migliorano.
• Confusion Matrix (Regressione Logistica):
    ◦ Visualizzata per comprendere le performance del modello.
    ◦ Esempio di valori estratti: TP = 100, TN = 100, FP = 0, FN = 0 (questi sono valori specifici estratti dalla matrice, che possono variare in base all'esecuzione).
    ◦ Ratio True/False: Il rapporto tra predizioni corrette (True) e predizioni errate (False) è stato calcolato come 67% (dato da (True Positives + True Negatives) / Totale).
2. Decision Tree
• Un modello DecisionTreeClassifier è stato addestrato per la classificazione della qualità delle mele.
• Analisi dei Criteri di Suddivisione (criterion): Sono stati testati i criteri 'gini' ed 'entropy' per la suddivisione dei rami.
    ◦ Conclusione: Nel caso specifico, il criterio entropy ha offerto le migliori performance.
• Confusion Matrix (Decision Tree):
    ◦ Visualizzata per comprendere le performance.
    ◦ Conclusione sul Confronto: Con il Decision Tree, ci sono stati meno errori rispetto alla Logistic Regression, e il rapporto tra predizioni corrette e errate è risultato maggiore (71% vs 67%), indicando una migliore performance.
• Feature Importance: Sono state calcolate e stampate le importanze delle feature, indicando quali caratteristiche delle mele sono state più rilevanti per il modello Decision Tree nella classificazione.
🧩 K-Means Clustering
Questa sezione esplora l'algoritmo di clustering non supervisionato K-Means.
1. Preparazione del Dataset: La label 'Quality' viene rimossa dal dataset classification_dataset per creare clustering_dataset, poiché K-Means non utilizza la label.
2. Clustering con n_clusters=2:
    ◦ Il modello KMeans è stato addestrato con due cluster, assumendo che le mele siano "buone" o "cattive".
    ◦ Vengono visualizzati i centroidi dei cluster.
    ◦ Una nuova colonna 'Cluster' viene aggiunta al dataset, indicando a quale cluster è stata assegnata ciascuna mela.
    ◦ È possibile effettuare una predizione per una mela specifica (data in input tramite ID), che indicherà se è "buona" o "cattiva".
3. Clustering con n_clusters=3:
    ◦ Viene esplorato lo scenario con tre cluster per verificare la possibilità di una qualità intermedia.
    ◦ Il modello KMeans viene addestrato con tre cluster.
    ◦ Vengono visualizzati i centroidi dei cluster.
    ◦ Anche qui, una nuova colonna 'Cluster' viene aggiunta.
    ◦ Effettuando una predizione, il modello può ora classificare una mela come "buona", "cattiva" o "di qualità intermedia".
⏳ Analisi delle Serie Temporali
L'ultima parte del progetto si concentra sulle serie temporali, utilizzando nuovamente il dataset originale regression_raw_dataset. L'obiettivo è analizzare come il gross income evolve nel tempo.
1. Creazione del Dataset Time Series: Viene creato un nuovo dataset, timeseries_dataset, contenente solo le colonne 'Date' e 'gross income'.
2. Regressione Lineare su Serie Temporali:
    ◦ La colonna 'Date' viene convertita in formato datetime e suddivisa in feature numeriche 'Day', 'Month', 'Year' per essere utilizzata nel modello.
    ◦ Il dataset viene diviso in training (80%) e test (20%).
    ◦ Un modello LinearRegression viene addestrato sulla serie temporale.
    ◦ Metriche di Valutazione:
        ▪ Mean Squared Error (MSE): 55.77.
        ▪ Mean Absolute Error (MAE): 5.58.
    ◦ Conclusione: I valori di MSE e MAE, sebbene possano sembrare non elevati, indicano che la regressione lineare spesso fallisce sulle serie temporali, suggerendo la necessità di modelli più performanti (come gli algoritmi specifici per serie temporali).
