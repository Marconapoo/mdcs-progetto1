from jacobi import jacobi, jacobi_sparse
from gauss_seidel import gauss_seidel
from gradiente_coniugato import gradiente_coniugato
from gradiente import gradiente
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread

if __name__ == '__main__':
    # Esempio
    A = np.array([[3, -1, 0, 0, 0, 0, 0, 0, 0], 
                [-1, 3, -1, 0, 0, 0, 0, 0, 0],
                [0, -1, 3, -1, 0, 0, 0, 0, 0],
                [0, 0, -1, 3, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, 3, -1, 0, 0, 0],
                [0, 0, 0, 0, -1, 3, -1, 0, 0],
                [0, 0, 0, 0, 0, -1, 3, -1, 0],
                [0, 0, 0, 0, 0, 0, -1, 3, -1],
                [0, 0, 0, 0, 0, 0, 0, -1, 3]], dtype=float)

    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    x0 = np.zeros_like(b)

    c = mmread('matrixes/spa1.mtx').tocsr()
    d = np.ones(c.shape[0])
    x0 = np.zeros_like(d)

    
    sol_j, iterations_j, error_j, time_j = jacobi_sparse(c, d, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_j)
    print("Errore: ", error_j)
    print("Numero iterazioni: ", iterations_j)
    print("Tempo di calcolo: ", time_j)

    

    sol_g, iterations_g, error_g, time_g = gauss_seidel(c, d, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_g)
    print("Errore: ", error_g)
    print("Numero iterazioni: ", iterations_g)
    print("Tempo di calcolo: ", time_g)

    sol_mg, iterations_mg, error_mg, time_mg = gradiente(c, d, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_mg)
    print("Errore: ", error_mg)
    print("Numero iterazioni: ", iterations_mg)
    print("Tempo di calcolo: ", time_mg)

    sol_gc, iterations_gc, error_gc, time_gc = gradiente_coniugato(c, d, x0, tol=1e-10)

    print("Soluzione approssimata: ", sol_gc)
    print("Errore: ", error_gc)
    print("Numero iterazioni: ", iterations_gc)
    print("Tempo di calcolo: ", time_gc)
    
    """
    sol_j, iterations_j, error_j, time_j = jacobi(A, b, x0, tol=1e-10)
    sol_gs, iterations_gs, error_gs, time_gs = gauss_seidel(A, b, x0, tol=1e-10)
    sol_gc, iterations_gc, error_gc, time_gc = gradiente_coniugato(A, b, x0, tol=1e-10)
    sol_g, iterations_g, error_g, time_g = gradiente(A, b, x0, tol=1e-10)
    


    print("METODO DI JACOBI:")
    print(f"Soluzione approssimata: {sol_j}\nerrore: {error_j}\nnumero iterazioni: {iterations_j}\ntempo di calcolo: {time_j}")
    print("METODO DI GAUSS SEIDEL:")
    print(f"Soluzione approssimata: {sol_gs}\nerrore: {error_gs}\nnumero iterazioni: {iterations_gs}\ntempo di calcolo: {time_gs}")
    print("METODO DEL GRADIENTE:")
    print(f"Soluzione approssimata: {sol_g}\nerrore: {error_g}\nnumero iterazioni: {iterations_g}\ntempo di calcolo: {time_g}")
    print("METODO DEL GRADIENTE CONIUGATO:")
    print(f"Soluzione approssimata: {sol_gc}\nerrore: {error_gc}\nnumero iterazioni: {iterations_gc}\ntempo di calcolo: {time_gc}")
    """