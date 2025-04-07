import numpy as np
from scipy.sparse import tril, csr_array 
import time

def jacobi(A, b, x0, tol, maxIter=20000):

    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    D = np.diag(np.diag(A))
    B = D - A
    D_inv = np.diag(1 / np.diag(A))

    x_old = x0.astype(float)
    x_new = x_old + 1.0
    k = 0
    error = 1.0
    start_time = time.time()

    while error > tol and k < maxIter:
        x_old = x_new.copy()
        x_new = D_inv @ (B @ x_old + b)
        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        k += 1

    end_time = time.time()
    total_time = end_time - start_time
    return x_new, k, error, 

def jacobi_sparse(A, b, x0, tol, maxIter=20000):
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")
    if A.shape[0] != len(b) or A.shape[0] != len(x0):
        raise ValueError("Incompatible dimensions")

    D = A.diagonal()                        
    if np.any(D == 0):
        raise ZeroDivisionError("Zero found on diagonal; Jacobi method won't work.")

    D_inv = 1.0 / D                         
    R = A.copy()
    R.setdiag(0)                            
    x_old = x0.astype(float)
    x_new = np.zeros_like(x_old)
    k = 0
    error = 1.0
    start_time = time.time()

    while error > tol and k < maxIter:
        x_new = D_inv * (b - R @ x_old)    
        error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
        x_old = x_new.copy()
        k += 1

    end_time = time.time()
    total_time = end_time - start_time

    return x_new, k, error, total_time