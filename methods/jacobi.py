import numpy as np
from scipy.sparse import tril, csr_array 
import time
from utils.matrix_utils import validate_matrix

def jacobi(A, b, x0, tol, maxIter=20000):
    """
    Risolve il sistema lineare Ax = b utilizzando il metodo iterativo di Jacobi.
    
    Parametri:
    ----------
    A : array_like o sparse matrix
        Matrice dei coefficienti
    b : array_like
        Vettore dei termini noti
    x0 : array_like
        Approssimazione iniziale della soluzione
    tol : float
        Tolleranza per il criterio di arresto (errore relativo)
    maxIter : int, opzionale
        Numero massimo di iterazioni permesse (default: 20000)
    
    Returns:
    --------
    x_new : ndarray
        Approssimazione della soluzione
    k : int
        Numero di iterazioni eseguite
    error : float
        Errore relativo dell'approssimazione finale
    total_time : float
        Tempo di esecuzione in secondi
    """

    # Estrazione degli elementi diagonali della matrice A
    D = A.diagonal()                        

    # Calcolo del reciproco degli elementi diagonali per l'iterazione di Jacobi
    # x^(k+1) = D^(-1) * (b - R * x^(k))
    D_inv = 1.0 / D                         
    
    # Creazione di una copia di A che diventerà la matrice R = A - D
    R = A.copy()
    
    # Azzeramento degli elementi diagonali per ottenere R
    R.setdiag(0)                            
    
    # Inizializzazione dei vettori soluzione
    x_old = x0.astype(float)  # Converte x0 in float
    x_new = np.zeros_like(x_old)
    
    # Inizializzazione contatori
    k = 0         # Contatore iterazioni
    error = 1.0   # Errore iniziale arbitrario > tol
    
    start_time = time.time()

    while error > tol and k < maxIter:

        x_new = D_inv * (b - R @ x_old)    
        
        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        
        x_old = x_new.copy()
        
        k += 1

    end_time = time.time()
    total_time = end_time - start_time

    return x_new, k, error, total_time