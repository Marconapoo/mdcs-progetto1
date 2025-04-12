import numpy as np
from scipy.sparse import isspmatrix, isspmatrix_csr, csr_matrix
import warnings

def validate_matrix(A, b, x0, method='all'):
    """
    Valida una matrice A, un vettore dei termini noti b e una soluzione iniziale x0
    per verificare se sono compatibili con il metodo iterativo specificato.
    
    Parameters:
    -----------
    A : array_like or sparse matrix
        Matrice del sistema lineare
    b : array_like
        Vettore dei termini noti
    x0 : array_like
        Soluzione iniziale
    method : str
        Metodo iterativo da usare ('jacobi', 'gauss_seidel', 'gradiente', 'gradiente_coniugato' o 'all')
    
    Returns:
    --------
    bool
        True se la matrice è valida per il metodo specificato, False altrimenti
    str
        Messaggio di errore o avviso
    """
    
    # Controlli di base
    M, N = A.shape
    if M != N:
        return False, "Matrix A must be square"
    
    if M != len(b) or M != len(x0):
        return False, "Incompatible dimensions"
    
    # Controllo elementi diagonali non nulli (per Jacobi e Gauss-Seidel)
    if method in ['jacobi', 'gauss_seidel', 'all']:
        diag_elements = A.diagonal()
        if np.any(diag_elements == 0):
            return False, "Zero diagonal elements found. Jacobi and Gauss-Seidel methods require non-zero diagonal elements"
    
    # Controlli specifici per metodo
    if method in ['jacobi', 'all']:
        # Verifica criterio di convergenza per Jacobi (diagonale dominante)
        if not is_diagonally_dominant(A):
            warnings.warn("Matrix is not diagonally dominant. Jacobi method might not converge.")
    
    if method in ['gauss_seidel', 'all']:
        # Verifica criterio di convergenza per Gauss-Seidel
        if not is_diagonally_dominant(A):
            warnings.warn("Matrix is not diagonally dominant. Gauss-Seidel method might not converge.")
    
    if method in ['gradiente_coniugato', 'gradiente', 'all']:
        # Verifica che la matrice sia simmetrica e definita positiva
        is_symm = is_symmetric(A)
        if not is_symm:
            warnings.warn("Matrix is not symmetric. Gradient method requires symmetric matrices.")
            
        # Test se la matrice è definita positiva (costoso, solo se la matrice è simmetrica)
        if is_symm and not is_positive_definite(A):
            warnings.warn("Matrix might not be positive definite. Gradient method requires positive definite matrices.")
    
    return True, "Matrix is valid for the specified method"

def is_diagonally_dominant(A):
    """
    Verifica se la matrice è diagonalmente dominante.
    """
    diag = np.abs(A.diagonal())
    row_sums = np.array([np.sum(np.abs(A[i].data)) - np.abs(A[i, i]) 
                         for i in range(A.shape[0])])
    return np.all(diag >= row_sums)

def is_symmetric(A, tol=1e-8):
    """
    Verifica se la matrice è simmetrica con una tolleranza.
    """
    if A.shape[0] != A.shape[1]:
        return False
    
    # Per matrici sparse, verifichiamo elemento per elemento
    if isspmatrix(A):
        diff = (A - A.T)
        return np.max(np.abs(diff.data)) < tol if diff.nnz > 0 else True
    else:
        return np.allclose(A, A.T, rtol=tol)

def is_positive_definite(A, tol=1e-8):
    """
    Verifica se la matrice è definita positiva.
    Per matrici di grandi dimensioni, usa il test di Cholesky invece di calcolare gli autovalori.
    """
    try:
        # Per matrici sparse di grandi dimensioni, questo è più efficiente
        if isspmatrix(A) and A.shape[0] > 1000:
            from scipy.sparse.linalg import spsolve
            n = A.shape[0]
            # Tenta di risolvere Ax = b con b > 0
            b = np.ones(n)
            x = spsolve(A, b)
            # Se tutti gli elementi di x sono positivi, è probabile che A sia definita positiva
            return np.all(x > 0)
        else:
            # Per matrici più piccole, usiamo la fattorizzazione di Cholesky
            from scipy.linalg import cholesky
            cholesky(A.toarray() if isspmatrix(A) else A)
            return True
    except:
        return False

