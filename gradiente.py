import numpy as np

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
    
    i = 0
    err = 1.0
    x_old = x0.astype(float)

    while i < nmax and err > tol:
        residual = b - A @ x_old
        step = residual @ residual / (residual @ (A @ residual))
        x_new = x_old + step * residual
        err = np.linalg.norm(b - A @ x_new)/np.linalg.norm(x_new)
        x_old = x_new
        i += 1
    
    return x_new, i, err