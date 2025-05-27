def print_matrix(matrix, title):
    print(title)
    for row in matrix:
        print("[", end="")
        for elem in row:
            print(f"{elem:10.6f}", end=" ")
        print("]")
    print()


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

    print_matrix(L, "Нижняя треугольная матрица L:")
    print_matrix(U, "Верхняя треугольная матрица U:")
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


# Пример использования
A = [
    [9.0, -5.0, -6.0, 3.0],
    [1.0, -7.0, 1.0, 0.0],
    [3.0, -4.0, 9.0, 0.0],
    [6.0, -1.0, 9.0, 8.0]
]
b = [-8.0, 38.0, 47.0, -8.0]

# Выполняем LU-разложение один раз
perm, L, U, swap_count = lu_decomposition(A)

print("Решение СЛАУ:")
x = solve_slau(A, b, perm, L, U)
print("x =", [round(val, 6) for val in x])

print("\nОпределитель матрицы A:")
det = determinant(U, swap_count)
print("det(A) =", round(det, 6))

print("\nОбратная матрица A^{-1}:")
inv_A = inverse_matrix(A, perm, L, U)
print_matrix(inv_A, "Обратная матрица A^{-1}:")

identity = matrix_multiply(A, inv_A)
print("\nПроверка A * A^{-1} = I:")
print_matrix(identity, "Результат умножения A * A^{-1}:")