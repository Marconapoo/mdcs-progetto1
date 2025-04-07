import numpy as np
import time

def gradiente_coniugato(A, b, x0, tol, maxIter=20000):

    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    #if not np.allclose(A, A.T):
     #   raise ValueError("Matrix A must be symmetric")
    
    #eigvals = np.linalg.eigvalsh(A)
    #if np.any(eigvals <= 0):
     #   raise ValueError("Matrix A must be positive definite")

    r = b - A @ x0
    p = r.copy()
    rsold = np.dot(r, r)

    k = 0
    error = 1.0

    start_time = time.time()
    
    while error > tol and k < maxIter:
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x0 += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        error = np.linalg.norm(A @ x0 - b) / np.linalg.norm(b)
        k += 1
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    end_time = time.time()
    total_time = end_time - start_time
    return x0, k, error, total_time
