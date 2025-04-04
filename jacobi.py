import numpy as np
import time
from scipy.io import mmread

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
    return x_new, k, error, total_time

# Esempio
A = np.array([[3, -1, 0, 0, 0, 0, 0, 0, 0], 
              [-1, 3, -1, 0, 0, 0, 0, 0, 0],
              [0, -1, 3, -1, 0, 0, 0, 0, 0],
              [0, 0, -1, 3, -1, 0, 0, 0, 0],
              [0, 0, 0, -1, 3, -1, 0, 0, 0],
              [0, 0, 0, 0, -1, 3, -1, 0, 0],
              [0, 0, 0, 0, 0, -1, 3, -1, 0],
              [0, 0, 0, 0, 0, 0, -1, 3, -1],
              [0, 0, 0, 0, 0, 0, 0, -1, 3]], dtype=float)

b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
x0 = np.zeros_like(b)

sol, iterations, error, timen = jacobi(A, b, x0, tol=1e-10)

print("METODO DI JACOBI:")
print(f"Soluzione approssimata: {sol}\nerrore: {error}\nnumero iterazioni: {iterations}\ntempo di calcolo: {timen}")
c = mmread('vem1.mtx')
print(c)

