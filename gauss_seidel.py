import numpy as np

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
    i = 0

    while np.linalg.norm(x_new - x_old) > tol and i < nmax:
        x_old = x_new
        rhs = b - B @ x_old
        x_new = np.linalg.solve(L, rhs)
        i += 1
    
    err = np.linalg.norm(x_new - x_old, ord=np.inf)
    return x_new, i, err

