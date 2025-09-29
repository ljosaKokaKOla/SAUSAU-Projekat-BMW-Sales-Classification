import matplotlib.pyplot as plt
import seaborn as sbn

#Glavna funkcija za iscrtavanje u pocetku,pokusavao da otkrijem kako da
#obradim anomalije u dataframeu
def scatter_visualize(df,column_name):
    if column_name not in df.columns:
        print("Column name not found")
        return

    plt.figure(figsize=(10,6))
    plt.scatter(df.index, df[column_name], alpha = 0.6, color = 'blue',edgecolors='black')
    plt.title(column_name)
    plt.xlabel(df.index)
    plt.ylabel(column_name)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#def plot_correlation_matrix(df, title ='Korelaciona matrica atributa'):

   #correlation_matrix = df.corr()

   #plt.figure(figsize=(20,18))
   #sbn.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=.5)
   #plt.title(title,fontsize=20)
   #plt.tight_layout()
   #plt.show(block=True)




