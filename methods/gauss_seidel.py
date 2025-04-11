import numpy as np
from scipy.sparse import tril
from scipy.sparse.linalg import spsolve
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import time

def gauss_seidel(A, b, x0, tol, nmax=20000):
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    L = tril(A)
    B = A - L

    x_old = x0.astype(float)
    x_new = x_old + 1.0
    k = 0
    error = 1.0

    start_time = time.time()
    while error > tol and k < nmax:
        x_old = x_new
        rhs = b - B @ x_old
        x_new = triang_inf_sparse(L, rhs)
        #x_new = spsolve(L, rhs)
        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        k += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    return x_new, k, error, total_time


def triang_inf(L, b):
    M, N = L.shape

    x = np.zeros(M)
    
    if M != N:
        print("Matrix L is not a square matrix")
        return x
    
    if not np.allclose(L, np.tril(L)):
        print("Matrix L is not a lower triangular matrix")
        return x
    
    x[0] = b[0] / L[0, 0]
    
    for i in range(1, N):
        somma = L[i, :i] @  x[:i]
        x[i] = (b[i] - somma) / L[i, i]
    
    return x

def triang_inf_sparse(L, b):
    """
    Risolve L x = b con forward substitution dove L è triangolare inferiore e sparsa.
    """
    if not isspmatrix_csr(L):
        L = L.tocsr()

    M, N = L.shape
    if M != N:
        raise ValueError("Matrix L must be square")
    if len(b) != M:
        raise ValueError("Incompatible dimension for vector b")
    
    x = np.zeros(M)

    for i in range(M):
        row_start = L.indptr[i]
        row_end = L.indptr[i + 1]
        sum_ = 0.0
        diag = None

        for idx in range(row_start, row_end):
            j = L.indices[idx]
            val = L.data[idx]
            if j < i:
                sum_ += val * x[j]
            elif j == i:
                diag = val

        if diag is None or diag == 0:
            raise ZeroDivisionError(f"Zero diagonal element found at row {i}")

        x[i] = (b[i] - sum_) / diag

    return x