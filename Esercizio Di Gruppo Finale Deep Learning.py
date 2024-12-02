# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Caricare i dataset
file_performance = "UCL_AllTime_Performance_Table.csv"  # Percorso al file
file_finals = "UCL_Finals_1955-2023.csv"  # Percorso al file

# Leggi i dati dai file CSV
df1 = pd.read_csv(file_performance)
df2 = pd.read_csv(file_finals)

# Mostra le prime righe dei dataset
df1.head()
df2.head()

# Rimuovi le colonne non necessarie
df1 = df1.drop(columns=['#', 'Pt.'])

# Rinomina le colonne
df1 = df1.rename(columns={'M.': 'Match_Played', 'W': 'Wins', 
                          'D': 'Draw', 'L': 'Losses', 'goals': 'Goals', 
                          'Dif': 'Goal_Difference' })

# Splitta la colonna 'Goals' in 'Goals_Scored' e 'Goals_Against'
df1[['Goals_Scored', 'Goals_Against', 'Temp']] = df1['Goals'].str.split(':', expand=True)

# Converti 'Goals_Scored' in tipo numerico
df1['Goals_Scored'] = df1['Goals_Scored'].astype(int)

# Rimuovi le colonne non necessarie dopo lo split
df1.drop(columns=['Goals', 'Goals_Against', 'Temp'], inplace=True)

# Rimuovi le righe con meno di 10 partite giocate
df1.drop(df1[df1['Match_Played'] <= 10].index, inplace=True)

# Calcola le percentuali di vittorie, pareggi e sconfitte
df1['Win_Rate'] = df1['Wins'] / df1['Match_Played'] * 100
df1['Draw_Rate'] = df1['Draw'] / df1['Match_Played'] * 100
df1['Loss_Rate'] = df1['Losses'] / df1['Match_Played'] * 100

# Definiamo la variabile target (ad esempio "Wins" per determinare se una squadra ha vinto o meno)
df1['Victory'] = df1['Wins'].apply(lambda x: 1 if x > 0 else 0)  # 1 per vittoria, 0 per nessuna vittoria

# Definiamo le feature (caratteristiche utilizzate per la previsione)
features = ['Match_Played', 'Win_Rate', 'Draw_Rate', 'Loss_Rate', 'Goal_Difference', 'Goals_Scored']

# Creiamo il set di dati per l'addestramento
X = df1[features]
y = df1['Victory']

# Suddividiamo il dataset in training set e test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modello Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Allena il modello sui dati di addestramento
rf.fit(X_train, y_train)

# Effettua previsioni sui dati di test
y_pred = rf.predict(X_test)

# Calcola la precisione sul test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy:.4f}')

# Report di classificazione
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation per valutare l'overfitting
cross_val_score_rf = cross_val_score(rf, X, y, cv=5)  # Cross-validation a 5 pieghe
print("\nCross-validation scores:")
print(cross_val_score_rf)
print(f"Mean CV accuracy: {cross_val_score_rf.mean():.4f}")
print(f"Standard deviation of CV accuracy: {cross_val_score_rf.std():.4f}")

# Verifica se ci sono segnali di overfitting (se la precisione sui dati di training è significativamente più alta di quella sui dati di test)
train_accuracy = rf.score(X_train, y_train)
print(f'Training accuracy: {train_accuracy:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Grafico delle importanze delle feature (per vedere quali variabili sono più influenti)
feature_importances = pd.Series(rf.feature_importances_, index=features)
feature_importances = feature_importances.sort_values(ascending=False)

# Visualizza le importanze delle feature
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Estrai i dati per i vincitori e le stagioni
df = df2[['Winners', 'Season']]

df.head()

# Grafico a barre per il numero di vittorie per ciascun vincitore
winners_count = df2['Winners'].value_counts()

winners = winners_count.index
counts = winners_count.values  
plt.figure(figsize=(10, 6))  
plt.bar(winners, counts, color='skyblue')  # Crea il grafico a barre
plt.xlabel('Winners')  
plt.ylabel('Count')    
plt.title('Number of Wins per Winner')  
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()
