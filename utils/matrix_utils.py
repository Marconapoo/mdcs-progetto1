import numpy as np

def validate_matrix(A, b, x, method):
    M, N = A.shape
    if M != N:
        return False, "La matrice A non è quadrata."
    if b.shape[0] != M:
        return False, "Il vettore b ha dimensioni incompatibili con A."
    if x.shape[0] != N:
        return False, "Il vettore x ha dimensioni incompatibili con A."
    
    if method == "jacobi" or method == "gauss_seidel":
        if is_diagonally_dominant(A):
            return True, ""
        else:
            return True, f"La matrice A non è diagonalmente dominante, il metodo di {method} potrebbe non convergere."
    
    if method == "gradiente_coniugato" or method == "gradiente":
        if is_spd(A):
            return True, ""
        else:
            return False, "La matrice A non è SPD (simmetrica definita positiva), il metodo non può essere applicato."
        

def is_symmetric(A):
    return (A != A.T).nnz == 0

def is_diagonally_dominant(X):
    D = np.abs(X.diagonal())

    abs_X = np.abs(X)
    row_sums = np.array(abs_X.sum(axis=1)).flatten()

    S = row_sums - D
    if np.all(D > S):
        return True
    else:
        return False


def is_spd(A):

    if not is_symmetric(A):
        return False
    
    N = A.shape[0]
    if N < 100000:
        from scipy.linalg import cholesky
        try:
            cholesky(A.todense())
            return True
        except Exception as e:
            return False
    else:
        from scipy.sparse.linalg import eigsh
        min_eigenvalue = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
        return min_eigenvalue > 0
    


