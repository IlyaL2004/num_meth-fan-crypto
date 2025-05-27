import numpy as np
import math

def find_max_elem(A):
    max_i = 1
    max_j = 0
    max_elem = abs(A[1][0])
    n_a = A.shape[0]
    for i in range(n_a):
        for j in range(i + 1, n_a):
            if abs(A[i][j]) > max_elem:
                max_elem = abs(A[i][j])
                max_i = i
                max_j = j
    return max_i, max_j


def termination_criterion(A):
    n_a = A.shape[0]
    sum_squares = 0
    for i in range(n_a):
        for j in range(i + 1, n_a):
            sum_squares = sum_squares + (pow(A[i][j], 2))
    sqrt_sum_squares = pow(sum_squares, 0.5)
    return sqrt_sum_squares


def rotation_method(A, eps):
    n_A = A.shape[0]
    U = np.eye(n_A)
    eigenvectors = np.eye(n_A)

    while True:
        if termination_criterion(A) < eps:
            break
        max_i, max_j = find_max_elem(A)
        if A[max_i][max_i] == A[max_j][max_j]:
            phi = math.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A[max_i][max_j] / (A[max_i][max_i] - A[max_j][max_j]))
        U = np.eye(n_A)
        U[max_i][max_j] = -np.sin(phi)
        U[max_j][max_i] = np.sin(phi)
        U[max_i][max_i] = np.cos(phi)
        U[max_j][max_j] = np.cos(phi)
        A = np.transpose(U) @ A @ U
        eigenvectors = eigenvectors @ U
    return A, eigenvectors


def print_ans(A, eigenvectors):
    diagonal_elements = np.diag(A)
    print("Собственные значения:")
    print(diagonal_elements)
    print("\nСобственные векторы:")
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]
        print(f"Собственный вектор {i + 1}:")
        print(eigenvector)

if __name__ == "__main__":
    A = np.array([[5, 5, 3],
                  [5, -4, 1],
                  [3, 1, 2]], dtype=float)

    eps = 1e-15
    A, eigenvectors = rotation_method(A, eps)

    print_ans(A, eigenvectors)