from jacobi import jacobi, jacobi_sparse
from gauss_seidel import gauss_seidel
from gradiente_coniugato import gradiente_coniugato
from gradiente import gradiente
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from grafici_relazione import genera_grafici

if __name__ == '__main__':

    A = mmread('matrixes/spa1.mtx').tocsr()
    b = np.ones(A.shape[0])
    x0 = np.zeros_like(b)

    
    sol_j, iterations_j, error_j, time_j = jacobi_sparse(A, b, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_j)
    print("Errore: ", error_j)
    print("Numero iterazioni: ", iterations_j)
    print("Tempo di calcolo: ", time_j)
    

    sol_gs, iterations_gs, error_gs, time_gs = gauss_seidel(A, b, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_gs)
    print("Errore: ", error_gs)
    print("Numero iterazioni: ", iterations_gs)
    print("Tempo di calcolo: ", time_gs)

    sol_mg, iterations_mg, error_mg, time_mg = gradiente(A, b, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_mg)
    print("Errore: ", error_mg)
    print("Numero iterazioni: ", iterations_mg)
    print("Tempo di calcolo: ", time_mg)

    sol_gc, iterations_gc, error_gc, time_gc = gradiente_coniugato(A, b, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_gc)
    print("Errore: ", error_gc)
    print("Numero iterazioni: ", iterations_gc)
    print("Tempo di calcolo: ", time_gc)

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