import numpy as np
import time

def gauss_seidel(A, b, x0, tol, nmax):
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    L = np.tril(A)
    B = A - L

    x_old = x0.astype(float)
    x_new = x_old + 1.0
    k = 0
    error = 1.0

    start_time = time.time()
    while error > tol and k < nmax:
        x_old = x_new
        rhs = b - B @ x_old
        x_new = triang_inf(L, rhs)
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

x_sol, iterations, error, gtime = gauss_seidel(A, b, x0, tol=1e-10, nmax=10000)
print("METODO DI GAUSS-SEIDEL:")
print(f"Soluzione approssimata: {x_sol}\nerrore: {error}\nnumero iterazioni: {iterations} \ntempo di calcolo: {gtime}")