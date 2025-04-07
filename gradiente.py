import numpy as np
from scipy.sparse.linalg import eigsh
import time 

def gradiente(A, b, x0, tol, nmax=2000000):
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    #if not np.allclose(A, A.T):
      #  raise ValueError("Matrix A must be symmetric")
    
    eigvals = eigsh(A)
    #if np.any(eigvals <= 0):
      #  raise ValueError("Matrix A must be positive definite")
    
    k = 0
    error = 1.0
    x_old = x0.astype(float)

    start_time = time.time()

    while k < nmax and error > tol:
        residual = b - A @ x_old
        step = residual @ residual / (residual @ (A @ residual))
        x_new = x_old + step * residual
        error = np.linalg.norm(b - A @ x_new)/np.linalg.norm(x_new)
        x_old = x_new
        k += 1
    
    end_time = time.time()
    total_time = end_time - start_time

    return x_new, k, error, total_time
