from methods.jacobi import jacobi
from methods.gauss_seidel import gauss_seidel
from methods.gradiente_coniugato import gradiente_coniugato
from methods.gradiente import gradiente
import matplotlib.pyplot as plt
import numpy as np

def genera_grafici(risultati):
    metodi = list(risultati.keys())
    
    iterazioni = [risultati[m]["n_iter"] for m in metodi]
    errori = [risultati[m]["errore"] for m in metodi]
    tempi = [risultati[m]["tempo"] for m in metodi]
    
    # Iterazioni
    plt.figure(figsize=(10, 6))
    plt.bar(metodi, iterazioni, color='skyblue')
    plt.ylabel('Numero di iterazioni')
    plt.title('Confronto numero di iterazioni')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Errore finale
    plt.figure(figsize=(10, 6))
    plt.bar(metodi, errori, color='salmon')
    plt.ylabel('Errore finale')
    plt.title('Confronto errore finale')
    plt.yscale('log')
    plt.grid(axis='y', which='both')
    plt.tight_layout()
    plt.show()
    
    # Tempo di esecuzione
    plt.figure(figsize=(10, 6))
    plt.bar(metodi, tempi, color='lightgreen')
    plt.ylabel('Tempo (secondi)')
    plt.title('Confronto tempo di esecuzione')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Confronto tra errore finale ed iterazioni totali
    plt.figure(figsize=(10, 6))
    for m in metodi:
        plt.scatter(risultati[m]["n_iter"], risultati[m]["errore"], label=m, s=100)
    plt.title('Errore finale vs. Iterazioni')
    plt.xlabel('Numero di iterazioni')
    plt.ylabel('Errore finale')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Confronto delle soluzioni (se disponibili)
    # Solo se tutte le soluzioni hanno la stessa dimensione
    lunghezze = [len(risultati[m]["soluzione"]) for m in metodi]
    if len(set(lunghezze)) == 1:
        plt.figure(figsize=(10, 6))
        for m in metodi:
            plt.plot(risultati[m]["soluzione"], marker='o', label=m)
        plt.title('Confronto delle soluzioni finali')
        plt.xlabel('Indice')
        plt.ylabel('Valore soluzione')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    else:
        print("Le soluzioni non hanno la stessa lunghezza: impossibile confrontarle graficamente.")

