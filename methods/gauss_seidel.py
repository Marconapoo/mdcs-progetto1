import numpy as np
from scipy.sparse import tril
import numpy as np
from scipy.sparse import isspmatrix_csr
import time
from utils.matrix_utils import validate_matrix



def gauss_seidel(A, b, x0, tol, nmax=20000):
    """
    Risolve il sistema lineare Ax = b utilizzando il metodo iterativo di Gauss-Seidel.
    
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
    nmax : int, opzionale
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

    valid, msg = validate_matrix(A, b, x0, method='gauss_seidel')
    if not valid:
        raise ValueError(msg)
    
    # Decomposizione della matrice A = L + (A-L)
    # L è la parte triangolare inferiore di A (inclusa la diagonale)
    L = tril(A)
    # B è il resto della matrice (parte triangolare superiore senza la diagonale)
    B = A - L


    x_old = x0.astype(float)  # Converte x0 in float 
    x_new = x_old.copy()
    k = 0                     # Contatore iterazioni
    error = 1.0               # Errore iniziale arbitrario > tol


    start_time = time.time()
    
    while error > tol and k < nmax:
        x_old = x_new
        
        rhs = b - B @ x_old
        
        x_new = triang_inf_sparse(L, rhs)

        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        
        k += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return x_new, k, error, total_time


def triang_inf_sparse(L, b):
    """
    Risolve L x = b con forward substitution dove L è triangolare inferiore e sparsa.
    
    Parametri:
    ----------
    L : sparse matrix
        Matrice triangolare inferiore sparsa
    b : array_like
        Vettore dei termini noti
        
    Returns:
    --------
    x : ndarray
        Soluzione del sistema lineare
        
    Note:
    -----
    La funzione implementa l'algoritmo di sostituzione in avanti (forward substitution)
    ottimizzato per matrici sparse nel formato CSR.
    """
    # Converti la matrice in formato CSR se non lo è già
    if not isspmatrix_csr(L):
        L = L.tocsr()

    # Verifica delle dimensioni
    M, N = L.shape
    if M != N:
        raise ValueError("Matrix L must be square")
    if len(b) != M:
        raise ValueError("Incompatible dimension for vector b")
    
    # Inizializza il vettore soluzione
    x = np.zeros(M)

    # Algoritmo di forward substitution
    for i in range(M):
        # Indici di inizio e fine della riga i nel formato CSR
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]
        sum_ = 0.0
        diag = None

        # Scansione degli elementi non-zero nella riga i
        for idx in range(row_start, row_end):
            j = L.indices[idx]      # Indice di colonna dell'elemento
            val = L.data[idx]       # Valore dell'elemento
            if j < i:
                # Somma dei prodotti L[i,j] * x[j] per j < i
                sum_ += val * x[j]
            elif j == i:
                # Memorizza l'elemento diagonale L[i,i]
                diag = val

        # Verifica che l'elemento diagonale non sia zero
        if diag is None or diag == 0:
            raise ZeroDivisionError(f"Zero diagonal element found at row {i}")

        # Calcolo della componente i-esima della soluzione
        # x[i] = (b[i] - sum(L[i,j] * x[j] per j < i)) / L[i,i]
        x[i] = (b[i] - sum_) / diag

    return x