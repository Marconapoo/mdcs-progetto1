import numpy as np
import time
from utils.matrix_utils import validate_matrix

def gradiente_coniugato(A, b, x0, tol, maxIter=20000):
    """
    Risolve il sistema lineare Ax = b utilizzando il metodo del gradiente coniugato.
    
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
        Numero massimo di iterazioni permesse (default: 20000)
    
    Returns:
    --------
    x0 : ndarray
        Approssimazione della soluzione
    k : int
        Numero di iterazioni eseguite
    error : float
        Errore relativo dell'approssimazione finale
    total_time : float
        Tempo di esecuzione in secondi
    """
    
    valid, msg = validate_matrix(A, b, x0, method='gradiente_coniugato')
    if not valid:
        raise ValueError(msg)

    # Calcolo del residuo iniziale: r = b - Ax_0
    r = b - A @ x0
    
    p = r.copy()
    
    rsold = np.dot(r, r)

    k = 0         # Contatore iterazioni
    error = 1.0   # Errore iniziale arbitrario > tol

    start_time = time.time()
    
    while error >= tol and k <= maxIter:

        Ap = A @ p
        
        den = np.dot(p, Ap)
        if den < 10e-15:
            break
        alpha = rsold / np.dot(p, Ap)
        
        x0 += alpha * p
        
        r -= alpha * Ap
        
        rsnew = np.dot(r, r)
        
        error = np.linalg.norm(A @ x0 - b) / np.linalg.norm(b)
        
        k += 1
        
        beta = rsnew / rsold
        
        p = r + beta * p
        
        rsold = rsnew

    end_time = time.time()
    total_time = end_time - start_time
    
    return x0, k, error, total_time
