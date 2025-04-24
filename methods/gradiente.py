import numpy as np
from scipy.sparse.linalg import eigsh
import time 
from utils.matrix_utils import validate_matrix

def gradiente(A, b, x0, tol, maxIter=20000):
    """
    Risolve il sistema lineare Ax = b utilizzando il metodo del gradiente.
    
    Parametri:
    ----------
    A : array_like
        Matrice dei coefficienti (deve essere simmetrica e definita positiva)
    b : array_like
        Vettore dei termini noti
    x0 : array_like
        Approssimazione iniziale della soluzione
    tol : float
        Tolleranza per il criterio di arresto (errore relativo)
    maxIter : int, opzionale
        Numero massimo di iterazioni permesse (default: 2000000)
    
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
    
    valid, msg = validate_matrix(A, b, x0, method='gradiente')
    if not valid:
        raise ValueError(msg)

    k = 0
    error = 1.0
    x_old = x0.astype(float)  # Converte x0 in float per evitare problemi con tipi interi
    x_new = x_old.copy()  # Inizializzazione arbitraria diversa da x_old

    start_time = time.time()

    while k <= maxIter and error >= tol:

        residual = b - A @ x_old
        Ar = A @ residual
        den = residual @ Ar
        step = residual @ residual / den
        x_new = x_old + step * residual
        
        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        
        x_old = x_new
        k += 1
    
    end_time = time.time()
    total_time = end_time - start_time

    return x_new, k, error, total_time
