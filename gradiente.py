import numpy as np
import time 

def gradiente(A, b, x0, tol, nmax):
    M, N = A.shape
    if M != N:
        raise ValueError("Matrix A must be square")
    if M != len(b) or M != len(x0):
        raise ValueError("Incompatible dimensions")
    
    if not np.allclose(A, A.T):
        raise ValueError("Matrix A must be symmetric")
    
    eigvals = np.linalg.eigvalsh(A)
    if np.any(eigvals <= 0):
        raise ValueError("Matrix A must be positive definite")
    
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

sol, iterations, error, total_time = gradiente(A, b, x0, tol=1e-10, nmax=10000)
print("METODO DEL GRADIENTE:")
print(f"Soluzione approssimata: {sol}\nerrore: {error}\nnumero iterazioni: {iterations}  \ntempo di calcolo: {total_time}")
