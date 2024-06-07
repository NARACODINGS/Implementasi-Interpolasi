import numpy as np
import matplotlib.pyplot as plt

# Fungsi Interpolasi Lagrange
def lagrange_interpolation(x, y, xi):
    n = len(x)
    yi = 0
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (xi - x[j]) / (x[i] - x[j])
        yi += y[i] * Li
    return yi

# Fungsi Interpolasi Newton
def newton_interpolation(x, y, xi):
    n = len(x)
    divided_diff = np.zeros((n, n))
    divided_diff[:,0] = y

    for j in range(1, n):
        for i in range(n-j):
            divided_diff[i,j] = (divided_diff[i+1,j-1] - divided_diff[i,j-1]) / (x[i+j] - x[i])

    yi = divided_diff[0,0]
    polynomial = 1.0
    for i in range(1, n):
        polynomial *= (xi - x[i-1])
        yi += divided_diff[0,i] * polynomial
    return yi

# Data yang diberikan
x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Testing Lagrange interpolation
print("Testing Lagrange Interpolation:")
for xi in x:
    yi_lagrange = lagrange_interpolation(x, y, xi)
    print(f"x = {xi}, interpolated y = {yi_lagrange}")

# Testing Newton interpolation
print("\nTesting Newton Interpolation:")
for xi in x:
    yi_newton = newton_interpolation(x, y, xi)
    print(f"x = {xi}, interpolated y = {yi_newton}")

# Plot interpolasi Lagrange
x_values = np.linspace(5, 40, 400)
y_values_lagrange = [lagrange_interpolation(x, y, xi) for xi in x_values]

plt.plot(x_values, y_values_lagrange, label='Lagrange Interpolation')
plt.scatter(x, y, color='red', label='Data Points')
plt.title('Interpolasi Lagrange')
plt.xlabel('Tegangan (Kg/mm^2)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.grid(True)
plt.show()

# Plot interpolasi Newton
y_values_newton = [newton_interpolation(x, y, xi) for xi in x_values]

plt.plot(x_values, y_values_newton, label='Newton Interpolation')
plt.scatter(x, y, color='red', label='Data Points')
plt.title('Interpolasi Newton')
plt.xlabel('Tegangan (Kg/mm^2)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.grid(True)
plt.show()

# Plot perbandingan interpolasi Lagrange dan Newton
plt.plot(x_values, y_values_lagrange, label='Lagrange Interpolation')
plt.plot(x_values, y_values_newton, label='Newton Interpolation', linestyle='--')
plt.scatter(x, y, color='red', label='Data Points')
plt.title('Interpolasi Lagrange vs Newton')
plt.xlabel('Tegangan (Kg/mm^2)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.grid(True)
plt.show()
