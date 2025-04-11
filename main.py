from jacobi import jacobi
from gauss_seidel import gauss_seidel
from gradiente_coniugato import gradiente_coniugato
from gradiente import gradiente
import numpy as np
from scipy.io import mmread
from grafici_relazione import genera_grafici
from tkinter.filedialog import askopenfilename
import time
import os
import sys


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    if not os.path.exists('matrixes'):
        os.makedirs('matrixes')
    print("Carica la matrice in formato .mtx. \n Il file deve essere presente nella cartella 'matrixes' e il nome del file deve essere in formato 'nome.mtx'.")
    filelocation = ""
    while(filelocation == ""):
        time.sleep(1)
        filelocation = askopenfilename(initialdir=os.getcwd() + "\matrixes", title="Seleziona il file .mxt", filetypes=[("Matrix Market files", "*.mtx")])
        if(filelocation == ""):
            print("Nessun file selezionato. Verifica che il file sia presente nella cartella 'matrixes' e che il nome del file sia in formato 'nome.mtx'.")

    file_name=filelocation.split("/")[-1]

    A = mmread(f'matrixes/{file_name}').tocsr()
    x0 = np.ones(A.shape[0])
    b = np.random.rand(A.shape[0])  


    sol_j, iterations_j, error_j, time_j = jacobi(A, b, x0, tol=1e-10)

    sol_gs, iterations_gs, error_gs, time_gs = gauss_seidel(A, b, x0, tol=1e-10)

    sol_mg, iterations_mg, error_mg, time_mg = gradiente(A, b, x0, tol=1e-10)

    sol_gc, iterations_gc, error_gc, time_gc = gradiente_coniugato(A, b, x0, tol=1e-10)


    print("METODO DI JACOBI:")
    print(f"Soluzione approssimata: {sol_j}\nerrore: {error_j}\nnumero iterazioni: {iterations_j}\ntempo di calcolo: {time_j}")
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
    