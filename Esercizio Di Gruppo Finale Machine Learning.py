
import pandas as pd
import matplotlib.pyplot as plt

# Caricare i dataset
file_performance = "UCL_AllTime_Performance_Table.csv"  # Percorso al file
file_finals = "UCL_Finals_1955-2023.csv" 

df1=pd.read_csv(file_performance)
df2=pd.read_csv(file_finals)

df1.head()
df2.head()

df1 = df1.drop(columns=['#', 'Pt.'])
df1 = df1.rename(columns={'M.':'Match_Played', 'W':'Wins',
                       'D':'Draw', 'L':'Losses', 'goals':'Goals', 
                        'Dif':'Goal_Difference' })

df1[['Goals_Scored', 'Goals_Against', 'Temp']] = df1['Goals'].str.split(':', expand=True)
df1['Goals_Scored'] = df1['Goals_Scored'].astype(int) 
df1.drop(columns=['Goals','Goals_Against', 'Temp'], inplace=True)

df1.head()

df1.drop(df1[df1['Match_Played'] <= 10].index, inplace=True)

df1.head()



df1['Win_Rate'] = df1['Wins'] / df1['Match_Played'] * 100
df1['Draw_Rate'] = df1['Draw'] / df1['Match_Played'] * 100
df1['Loss_Rate'] = df1['Losses'] / df1['Match_Played'] * 100
df1.head()


df1.set_index('Team')[['Match_Played', 'Wins', 'Draw', 'Losses']].head(10) \
.plot(kind='bar', figsize=(12,6))
plt.title('Matches Played, Wins, Draws, and Losses by Team')
plt.xlabel('Team')
plt.ylabel('Count')
plt.legend(title='Legend')
plt.show()

df1.set_index('Team')['Goal_Difference'].head(10) \
.plot(kind='bar', figsize=(12,6))
plt.title('Goal Difference by Team')
plt.xlabel('Team')
plt.ylabel('Goal_Difference')
plt.show()



df = df2[['Winners', 'Season']]

df.head()

plt.bar()

winners_count = df2['Winners'].value_counts()

winners = winners_count.index 
counts = winners_count.values  
plt.figure(figsize=(10, 6))  
plt.bar(winners, counts, color='skyblue')  
plt.xlabel('Winners')  
plt.ylabel('Count')    
plt.title('Number of Wins per Winner')  
plt.xticks(rotation=45)  
plt.tight_layout()  
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

l1=df1['Win_Rate']
l2=df1['Draw_Rate']
l3=df1['Loss_Rate']







