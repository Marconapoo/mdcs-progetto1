import numpy as np
import time

def gradiente_coniugato(A, b, x0, tol, maxIter=20000):
    n = A.shape[0]
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

sol, iterations, error, timen = gradiente_coniugato(A, b, x0, tol=1e-10)

print("METODO DEL GRADIENTE CONIUGATO:")
print(f"Soluzione approssimata: {sol}\nerrore: {error}\nnumero iterazioni: {iterations}\ntempo di calcolo: {timen}")