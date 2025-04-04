from jacobi import jacobi
from gauss_seidel import gauss_seidel
from gradiente_coniugato import gradiente_coniugato
from gradiente import gradiente
import matplotlib.pyplot as plt
import numpy as np

N = 100  # Sostituisci con il valore desiderato
tol = 1e-10
nmax = 100
nit_Jac_vect = {}
time_Jac_vect = {}
err_Jac_vect = {}
nit_GSe_vect = {}
time_GSe_vect = {}
err_GSe_vect = {}

nit_mg_vect = {}
time_mg_vect = {}
err_mg_vect = {}

for n in range(3, N+1, 3):
    print(n)
    A = np.diag(3 * np.ones(n))
    A -= np.diag(np.ones(n-1), -1) + np.diag(np.ones(n-1), 1)
    b = np.ones(n)
    x0 = np.random.rand(n)
    
    x_Grad, nit_Grad, err_Grad, time_Grad = gradiente_coniugato(A, b, x0, tol, nmax)
    x_Jac, nit_Jac, err_Jac, time_Jac = jacobi(A, b, x0, tol, nmax)
    x_GSe, nit_GSe, err_GSe, time_GSe = gauss_seidel(A, b, x0, tol, nmax)
    x_mg, nit_mg, err_mg, time_mg = gradiente(A, b, x0, tol, nmax)
    
    nit_Jac_vect[n] = nit_Jac
    time_Jac_vect[n] = time_Jac
    err_Jac_vect[n] = err_Jac
    
    nit_GSe_vect[n] = nit_GSe
    time_GSe_vect[n] = time_GSe
    err_GSe_vect[n] = err_GSe
    
    nit_mg_vect[n] = nit_mg
    time_mg_vect[n] = time_mg
    err_mg_vect[n] = err_mg

print("GS")
print(nit_GSe_vect)
print("Jacobi")
print(nit_Jac_vect)
print("Gradiente")
print(nit_mg_vect)
n_values = list(range(3, N+1, 3))

plt.figure(1)
plt.plot(n_values, [nit_Jac_vect[n] for n in n_values], '*', label='Jacobi')
plt.plot(n_values, [nit_GSe_vect[n] for n in n_values], 'ro', label='Gauss Seidel')
plt.plot(n_values, [nit_mg_vect[n] for n in n_values], 'k+', label='Metodo del gradiente')
plt.title('Number of iterations')
plt.legend(loc='upper left')
plt.grid()

plt.figure(2)
plt.plot(n_values, [time_Jac_vect[n] for n in n_values], '*', label='Jacobi')
plt.plot(n_values, [time_GSe_vect[n] for n in n_values], 'ro', label='Gauss Seidel')
plt.plot(n_values, [time_mg_vect[n] for n in n_values], 'k+', label='Metodo del gradiente')
plt.title('Time elapsed')
plt.legend(loc='upper left')
plt.grid()

plt.figure(3)
plt.plot(n_values, [err_Jac_vect[n] for n in n_values], '*', label='Jacobi')
plt.plot(n_values, [err_GSe_vect[n] for n in n_values], 'ro', label='Gauss Seidel')
plt.plot(n_values, [err_mg_vect[n] for n in n_values], 'k+', label='Metodo del gradiente')
plt.title('Final errors')
plt.legend(loc='upper left')
plt.grid()

plt.show()

