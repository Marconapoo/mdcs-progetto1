import numpy as np
import time

def jacobi(A, b, x0, tol, maxIter=10000):
    n = A.shape[0]
    D = np.diag(np.diag(A))
    B = D - A
    D_inv = np.diag(1 / np.diag(A))

    x_old = x0.copy()
    x_new = x_old + 1
    k = 0
    start_time = time.time()

    while np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) > tol and k < maxIter:
        x_old = x_new.copy()
        x_new = D_inv @ (B @ x_old + b)
        k += 1
    total_time = time.time() - start_time
    error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)
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
