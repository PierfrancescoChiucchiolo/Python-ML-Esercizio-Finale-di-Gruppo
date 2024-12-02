README.TXT


Partendo da una passione calcistica comune, tranne Pierfrancesco, abbiamo pensato di analizzare i dati dal 1955 a 2023 inerenti il campionato UEFA per trarne delle metriche di interesse.
Abbiamo caricato il dataset contenente le performance per ogni squadra e il campionato anno per anno della UEFA Champions League, chiamate rispettivamente:
UCL_AllTime_Performance_Table.csv
UCL_Finals_1955-2023.csv

I files sono dizionari in formato "Comma Separated Values", ovvero .csv, e contengono le seguenti chiavi:
#,Team,M.,W,D,L,goals,Dif,Pt.
Season,Country,Winners,Score,Runners-up,Country,Venue,Attendance,Notes

Si è deciso di rimuovere immediatamente le colonne relative alle chiavi # e Pt. siccome l'identificativo non è presente in alcuni valori come 112, 253, e Pt. invece è totalmente uguale alla colonna Dif
Poi si è deciso di rinominare alcune tables per migliore leggibilità come segue:
'M.'->'Match_Played'
'W' -> 'Wins'
'D' -> 'Draw'
'L' -> 'Losses'
'goals' ->'Goals'
'Dif' -> 'Goal_Difference'

Per una migliore analisi si ritiene degno di nota creare alcune colonne extra per meglio raffinare i dati come:

Win_Rate = percentuale tra Wins e Match_Played
Draw_Rate = percentuale tra Draw e Match_Played
Loss_Rate = percentuale tra Loss e Match_Played

Alcuni grafici illustrano la Top10 squadre e le loro vittorie/pareggi/sconfitte, oltre alla loro differenza reti


proof of concept di codice non ancora scritto per via di mancanza di tempo:

il dataset non ci offre dati relativi ad ogni match, ma solo delle finali
in base al numero di partite disputate possiamo normalizzare un indice che ci dice quanto probabile sia che la partita che si disputa coinvolga quella squadra
questo indice è correlato con l'indice di probabilità che una squadra si qualifichi, siccome più partite ha disputato più a lungo e spesso ha partecipato nel campionato
con questo indice possiamo estrarre 16 squadre e grazie ad esso come peso ne indichiamo la probabilità che sia qualificata e poi predire la squadra vincitrice del torneo

con questo tipo di inferenza sarebbe possibile per esempio prevedere i vincitori dei prossimi campionati anche basandoci sui dati dello scontro diretto in finale, oltre alle metriche che abbiamo estratto come il win/draw/loss rate


Il Gruppo ha provato anche una versione di analisi con Deep Learning, ma il Dataset non si è rivelato opportuno


Componenti del Gruppo:

Pierfrancesco Chiucchiolo
https://www.linkedin.com/in/pierfrancesco-chiucchiolo-66760021a/
https://github.com/PierfrancescoChiucchiolo/Corso-Python

Alessandro Perrone
https://www.linkedin.com/in/alessandro-perrone-a46618254/
https://github.com/alessandroperrone10/Python-e-ML

Domenico Iannizzi
https://www.linkedin.com/in/domenico-iannizzi-918687287/
https://github.com/domenicoiannizzi

Simone Buglione
https://www.linkedin.com/in/simonebuglione/
https://github.com/simonebuglione

Lorenzo Ciarlone
https://www.linkedin.com/in/lorenzociarlone/
https://github.com/Solace-hub
