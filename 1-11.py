"""def lu_decomposition(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)] # лист комплекеншен не использовать
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
        for i in range(k + 1, n):   # цикл поменять
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]
    return perm, L, U, swap_count"""

def lu_decomposition(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]  # создаем пустую матрицу L
    for i in range(n):
        L[i][i] = 1.0  # диагональные элементы L равны 1
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
            # меняем местами строки в U и L
            U[k], U[max_row] = U[max_row], U[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            # также меняем местами индексы в perm
            perm[k], perm[max_row] = perm[max_row], perm[k]
            swap_count += 1
        if U[k][k] == 0:
            raise ValueError("Матрица вырождена")

        # Сначала полностью вычисляем столбец L
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]

        # После этого вычисляем столбец U, используя уже вычисленный столбец L
        for i in range(k + 1, n):
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return perm, L, U, swap_count


def forward_substitution(L, b):
    """
    Решает систему уравнений Ly = b.
    Возвращает y.
    """
    n = len(L)
    y = [0] * n  # Создаем пустой вектор для y

    for i in range(n):
        s = 0  # Сумма для вычисления y[i]
        for j in range(i):  # Суммируем элементы до текущего
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) #/ L[i][i]  # Вычисляем y[i]

    return y

"""def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    return y"""


def backward_substitution(U, y):
    """
    Решает систему уравнений Ux = y.
    Возвращает x.
    """
    n = len(U)
    x = [0] * n  # Создаем пустой вектор для x

    for i in range(n - 1, -1, -1):  # Идем с конца, начиная с n-1
        s = 0  # Сумма для вычисления x[i]
        for j in range(i + 1, n):  # Суммируем элементы после текущего
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]  # Вычисляем x[i]

    return x


"""def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x"""

def solve_slau(A, b):
    perm, L, U, swap_count = lu_decomposition(A)
    b_perm = [b[perm[i]] for i in range(len(perm))]
    y = forward_substitution(L, b_perm)
    x = backward_substitution(U, y)
    return x

"""def determinant(A):
    perm, L, U, swap_count = lu_decomposition(A)
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    det *= (-1) ** swap_count # поменять определение знака
    return det"""


def determinant(A):
    perm, L, U, swap_count = lu_decomposition(A)
    det = 1.0

    # Считаем знак детерминанта через цикл
    sign = 1  # Начальный знак = +1 (по умолчанию)
    for i in range(swap_count):
        sign *= -1  # Каждый раз меняем знак на противоположный

    # Теперь вычисляем произведение диагональных элементов матрицы U
    for i in range(len(U)):
        det *= U[i][i]

    det *= sign  # Умножаем на знак, полученный через цикл
    return det


def inverse_matrix(A):
    perm, L, U, swap_count = lu_decomposition(A)
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

# Пример использования
A = [
    [9.0, -5.0, -6.0, 3.0],
    [1.0, -7.0, 1.0, 0.0],
    [3.0, -4.0, 9.0, 0.0],
    [6.0, -1.0, 9.0, 8.0]
]
b = [-8.0, 38.0, 47.0, -8.0]

print("Решение СЛАУ:")
x = solve_slau(A, b)
print("x =", [round(val, 6) for val in x])

print("\nОпределитель матрицы A:")
det = determinant(A)
print("det(A) =", round(det, 6))

print("\nОбратная матрица A^{-1}:")
inv_A = inverse_matrix(A)
for row in inv_A:
    print([round(elem, 6) for elem in row])

# Проверка обратной матрицы
def matrix_multiply(A, B):
    n = len(A)
    m = len(B[0])
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

identity = matrix_multiply(A, inv_A)
print("\nПроверка A * A^{-1} = I:")
for row in identity:
    print([round(elem, 2) for elem in row])