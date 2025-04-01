import numpy as np

def jacobi(A, b, x0, tol, maxIter=100):
    n = A.shape[0]
    x_old = x0.copy()
    # Calcolo di P: matrice diagonale
    P = np.diag(np.diag(A))
    # Calcolo di N: matrice ottenuta mettendo a 0 le entrate sulla diagonale di A e considerando l’opposto di tutte le altre entrate
    N = P - A
    P_inv = np.linalg.inv(P)
    k = 0
    x_new = x_old + 1.0

    while np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) > tol and k < maxIter:
        # Calcoliamo il nuovo vettore x^(k+1) = P^(−1) * ( b − N * x^(k))
        x_new = P_inv @ (b - N @ x_old)
        x_old = x_new  # Aggiorniamo x per la prossima iterazione
        k += 1
    error = np.linalg.norm(A @ x_new - b) / np.linalg.norm(b)

    # Se il ciclo termina senza convergenza
    return x_old, maxIter, error
def gradiente_coniugato(A, b, x0, tol, maxIter=20000):
    n = A.shape[0]
    r = b - A @ x0 #residuo iniziale
    p = r.copy() #direzione di ricerca
    rsold = np.dot(r, r)
    
    for k in range(maxIter):
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x0 += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        
        # Criterio di arresto
        if np.linalg.norm(A @ x0 - b) / np.linalg.norm(b) < tol:
            return x0, k + 1, True
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x0, maxIter, False

# Esempio di test
if __name__ == "__main__":
    a = np.array([[3, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-1, 3, -1, 0, 0, 0, 0, 0, 0],
                  [0, -1, 3, -1, 0, 0, 0, 0, 0],
                  [0, 0, -1, 3, -1, 0, 0, 0, 0],
                  [0, 0, 0, -1, 3, -1, 0, 0, 0],
                  [0, 0, 0, 0, -1, 3, -1, 0, 0],
                  [0, 0, 0, 0, 0, -1, 3, -1, 0],
                  [0, 0, 0, 0, 0, 0, -1, 3, -1],
                  [0, 0, 0, 0, 0, 0, 0, -1, 3]], dtype = float)

    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype = float)
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = float)

    sol, num_iter, error = jacobi(a, b, x0, 1e-10)
    print("METODO DI JACOBI:")
    print(f"Soluzione approssimata: {sol}, errore: {error}")
    """
    print("METODO DEL GRADIENTE CONIUGATO:")
    sol, num_iter, converged = gradiente_coniugato(a, b, x0, 1e-5)
    if converged:
        print(f"Soluzione trovata in {num_iter} iterazioni:")
    else:
        print(f"Non è stata raggiunta la convergenza in {num_iter} iterazioni.")
    print(f"Soluzione approssimata: {sol}")"
    """