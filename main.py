from methods.jacobi import jacobi
from methods.gauss_seidel import gauss_seidel
from methods.gradiente_coniugato import gradiente_coniugato
from methods.gradiente import gradiente
import numpy as np
from scipy.io import mmread
from utils.grafici_relazione import genera_grafici
from tkinter.filedialog import askopenfilename
import os
import sys


def load_matrix():
    np.set_printoptions(threshold=sys.maxsize)
    if not os.path.exists('matrixes'):
        os.makedirs('matrixes')
    print("Carica la matrice in formato .mtx. \n Il file deve essere presente nella cartella 'matrixes' e il nome del file deve essere in formato 'nome.mtx'.")
    filelocation = ""
    filelocation = askopenfilename(initialdir=os.getcwd() + "\matrixes", title="Seleziona il file .mxt", filetypes=[("Matrix Market files", "*.mtx")])
    if(filelocation == ""):
        print("Nessun file selezionato. Verifica che il file sia presente nella cartella 'matrixes' e che il nome del file sia in formato 'nome.mtx'.")

    file_name=filelocation.split("/")[-1]
    matrix = mmread(f'matrixes/{file_name}').tocsr()
    return matrix

def load_x_true(A):
    set_x_true = True
    while set_x_true:
        print("Vuoi caricare un vettore x soluzione esatta? (y/n)")
        choice = input()
        choice = choice.lower()
        if choice == 'y':
            x_true = load_matrix()
            x_true = x_true.toarray()
        elif choice == 'n':
            x_true = np.ones(A.shape[0])
            print("Il vettore x soluzione esatta contiene solo valori 1.")
            set_x_true = False
        else:
            print("Scelta non valida. Riprova.")
    return x_true

def load_b(A, x_true):
    set_b = True
    while set_b:
        print("Vuoi caricare un vettore b? (y/n)")
        choice = input()
        choice = choice.lower()
        if choice == 'y':
            b = load_matrix()
            b = b.toarray()
        elif choice == 'n':
            b = A @ x_true
            print("Il vettore b è stato calcolato come Ax.")
            set_b = False
        else:
            print("Scelta non valida. Riprova.")
    return b

if __name__ == '__main__':
    
    print("Benvenuto nel programma di risoluzione di sistemi lineari!")
    print("Caricamento della matrice A...")
    A = load_matrix()

    x_true = load_x_true(A)
    
    b = load_b(A, x_true)


    print("Seleziona l'esponente per la tolleranza, (4, 6, 8, 10, ...). \nATTENZIONE! Con un esponente troppo grande qualche metodo potrebbe non convergere:")
    set_exp = True
    while set_exp:
        try:
            exp = int(input())
            set_exp = False
        except ValueError:
            print("Errore: inserire un numero intero.")
    tol = 10 ** (-exp)
    print("Esecuzione dei metodi di risoluzione con tolleranza:", tol)

    x0 = np.random.randint(0, A.shape[0], size=A.shape[0])
    x0 = x0.astype(float)  

    sol_j, iterations_j, error_j, time_j = jacobi(A, b, x0, tol)

    sol_gs, iterations_gs, error_gs, time_gs = gauss_seidel(A, b, x0, tol)

    sol_mg, iterations_mg, error_mg, time_mg = gradiente(A, b, x0, tol)

    sol_gc, iterations_gc, error_gc, time_gc = gradiente_coniugato(A, b, x0, tol)


    print("METODO DI JACOBI:")
    print(f"Soluzione approssimata: {sol_j} \nerrore: {error_j}\nnumero iterazioni: {iterations_j}\ntempo di calcolo: {time_j}")
    print("METODO DI GAUSS SEIDEL:")
    print(f"Soluzione approssimata: {sol_gs}\nerrore: {error_gs}\nnumero iterazioni: {iterations_gs}\ntempo di calcolo: {time_gs}")
    print("METODO DEL GRADIENTE:")
    print(f"Soluzione approssimata: {sol_mg}\nerrore: {error_mg}\nnumero iterazioni: {iterations_mg}\ntempo di calcolo: {time_mg}")
    print("METODO DEL GRADIENTE CONIUGATO:")
    print(f"Soluzione approssimata: {sol_gc}\nerrore: {error_gc}\nnumero iterazioni: {iterations_gc}\ntempo di calcolo: {time_gc}")

    
    risultati = {"Jacobi": {"soluzione": sol_j, "n_iter": iterations_j, "errore": error_j, "tempo": time_j},
                 "Gauss-Seidel": {"soluzione": sol_gs, "n_iter": iterations_gs, "errore": error_gs, "tempo": time_gs},
                 "Gradiente": {"soluzione": sol_mg, "n_iter": iterations_mg, "errore": error_mg, "tempo": time_mg},
                 "Gradiente Coniugato": {"soluzione": sol_gc, "n_iter": iterations_gc, "errore": error_gc, "tempo": time_gc}}
    
    genera_grafici(risultati)

    relative_error_j = np.linalg.norm(sol_j - x_true) / np.linalg.norm(x_true)
    relative_error_gs = np.linalg.norm(sol_gs - x_true) / np.linalg.norm(x_true)
    relative_error_mg = np.linalg.norm(sol_mg - x_true) / np.linalg.norm(x_true)
    relative_error_gc = np.linalg.norm(sol_gc - x_true) / np.linalg.norm(x_true)

    print(f"Errore relativo Jacobi: {relative_error_j}")
    print(f"Errore relativo Gauss-Seidel: {relative_error_gs}")
    print(f"Errore relativo Gradiente: {relative_error_mg}")
    print(f"Errore relativo Gradiente Coniugato: {relative_error_gc}")
