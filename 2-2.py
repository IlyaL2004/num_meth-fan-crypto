import numpy as np

def lu_decomposition(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    perm = list(range(n))
    swap_count = 0

    for k in range(n):
        max_row = k
        max_val = abs(U[k][k])
        for i in range(k, n):
            if abs(U[i][k]) > max_val:
                max_row = i
                max_val = abs(U[i][k])
        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            perm[k], perm[max_row] = perm[max_row], perm[k]
            swap_count += 1
        if U[k][k] == 0:
            raise ValueError("Матрица вырождена")

        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]

        for i in range(k + 1, n):
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return perm, L, U, swap_count


def forward_substitution(L, b):
    n = len(L)
    y = [0] * n

    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s)

    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(i + 1, n):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]

    return x


def solve_slau(A, b, perm, L, U):
    b_perm = [b[perm[i]] for i in range(len(perm))]
    y = forward_substitution(L, b_perm)
    x = backward_substitution(U, y)
    return x


def determinant(U, swap_count):
    det = 1.0

    sign = 1
    for i in range(swap_count):
        sign *= -1

    for i in range(len(U)):
        det *= U[i][i]

    det *= sign
    return det


def inverse_matrix(A, perm, L, U):
    n = len(A)
    inv_A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e_perm = [0.0] * n
        for j in range(n):
            if perm[j] == i:
                e_perm[j] = 1.0
        y = forward_substitution(L, e_perm)
        x = backward_substitution(U, y)
        for j in range(n):
            inv_A[j][i] = x[j]
    return inv_A


def matrix_multiply(A, B):
    n = len(A)
    m = len(B[0])
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def f(x):
    x1, x2 = x
    return np.array([
        (x1 ** 2 + 16) * x2 - 64,  # f1(x1, x2)
        (x1 - 2) ** 2 + (x2 - 2) ** 2 - 16  # f2(x1, x2)
    ])

def jacobian(x):
    x1, x2 = x
    return np.array([
        [2 * x1 * x2, x1 ** 2 + 16],  # ∂f1/∂x1, ∂f1/∂x2
        [2 * (x1 - 2), 2 * (x2 - 2)]   # ∂f2/∂x1, ∂f2/∂x2
    ])

def newton_method_with_inverse(x_initial, eps=1e-6, max_iter=100):
    x = np.array(x_initial, dtype=float)
    iterations = 0
    for i in range(max_iter):
        iterations = i + 1
        J = jacobian(x)
        F = -f(x)

        perm, L, U, swap_count = lu_decomposition(J.tolist())
        inv_J = inverse_matrix(J.tolist(), perm, L, U)

        delta_x = np.dot(inv_J, F)

        delta_norm = np.max(np.abs(delta_x))

        x += delta_x
        if delta_norm < eps:
            break
    return x, iterations


def L_inf_norm(a):
    return np.max(np.abs(a))


def phi(x):
    x1, x2 = x
    return np.array([
        2 + np.sqrt(16 - (x2 - 2) ** 2),  # φ1(x1, x2)
        64 / (x1 ** 2 + 16)  # φ2(x1, x2)
    ])


def phi_jacobian(x):
    x1, x2 = x
    return np.array([
        [0, (2 - x2) / np.sqrt(16 - (x2 - 2) ** 2)],  # ∂φ1/∂x1, ∂φ1/∂x2
        [-128 * x1 / (x1 ** 2 + 16) ** 2, 0]  # ∂φ2/∂x1, ∂φ2/∂x2
    ])


def get_q(intervals):
    center = np.mean(intervals, axis=1)
    J = phi_jacobian(center)
    return np.max(np.sum(np.abs(J), axis=1))


def iteration_method(intervals, eps=1e-6, max_iters=1000):
    x_prev = np.mean(intervals, axis=1)
    q = get_q(intervals)

    for it in range(max_iters):
        x = phi(x_prev)
        delta = L_inf_norm(x - x_prev)
        if q / (1 - q) * delta < eps:
            return x, it + 1
        x_prev = x

    raise ValueError("Метод не сошёлся за максимальное число итераций")



x_initial = [5.0, 1.0]
solution, newton_iters = newton_method_with_inverse(x_initial)
print("Метод Ньютона:")
print(f"Решение: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}")
print(f"Количество итераций: {newton_iters}")


intervals = np.array([[0, 4], [0, 4]])
eps = 1e-6

solution, iter_iters = iteration_method(intervals, eps)
print("\nМетод простой итерации:")
print(f"Решение: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}")
print(f"Количество итераций: {iter_iters}")