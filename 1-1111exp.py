def vector_inf_norm(v):
    """Вычисляет L∞-норму для вектора (максимальный по модулю элемент)"""
    return max(abs(x) for x in v)


def matrix_inf_norm(M):
    """Вычисляет L∞-норму для матрицы (максимальная сумма модулей элементов строки)"""
    return max(sum(abs(x) for x in row) for row in M)


def prepare_system(A, b):
    """
    Подготавливает систему для итерационных методов:
    - Проверяет диагональные элементы.
    - Преобразует систему к виду x = alpha*x + beta.
    Возвращает alpha, beta и норму alpha.
    """
    n = len(A)
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError("Матрица содержит нулевые диагональные элементы")

    alpha = [[0.0 for _ in range(n)] for _ in range(n)]
    beta = [0.0 for _ in range(n)]

    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]

    alpha_norm = matrix_inf_norm(alpha)
    return alpha, beta, alpha_norm


def solve_iterative(A, b, eps):
    """Метод простых итераций"""
    alpha, beta, alpha_norm = prepare_system(A, b)
    cur_x = beta.copy()
    iteration = 0

    while True:
        prev_x = cur_x.copy()
        new_x = [0.0 for _ in range(len(A))]
        for i in range(len(A)):
            sum_val = 0.0
            for j in range(len(A)):
                sum_val += alpha[i][j] * prev_x[j]
            new_x[i] = sum_val + beta[i]
        cur_x = new_x
        delta = vector_inf_norm([cur_x[i] - prev_x[i] for i in range(len(cur_x))])
        iteration += 1

        error_estimate = (alpha_norm / (1 - alpha_norm)) * delta if alpha_norm < 1 else delta
        if error_estimate <= eps:
            return cur_x, iteration


def solve_seidel(A, b, eps):
    """Метод Зейделя"""
    alpha, beta, alpha_norm = prepare_system(A, b)
    cur_x = beta.copy()
    iteration = 0

    while True:
        prev_x = cur_x.copy()
        for i in range(len(A)):
            sum_val = 0.0
            for j in range(i):  # Используем обновленные значения
                sum_val += alpha[i][j] * cur_x[j]
            for j in range(i, len(A)):  # Используем старые значения
                sum_val += alpha[i][j] * prev_x[j]
            cur_x[i] = sum_val + beta[i]

        delta = vector_inf_norm([cur_x[i] - prev_x[i] for i in range(len(cur_x))])
        iteration += 1

        error_estimate = (alpha_norm / (1 - alpha_norm)) * delta if alpha_norm < 1 else delta
        if error_estimate <= eps:
            return cur_x, iteration


def matrix_vector_multiply(A, x):
    """Умножение матрицы на вектор"""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


if __name__ == "__main__":
    # Пример системы: 4x₁ + x₂ = 7; x₁ + 3x₂ = 10
    A = [[-23, -7, 5, 2], [-7, -21, 4, 9], [9, 5, -31, -8], [0, 1, -2, 10]]
    b = [-26, -55, -58, -24]
    eps = 1e-9

    print("Исходная система:")
    print(f"A = {A}")
    print(f"b = {b}\n")

    # Решение методом простых итераций
    try:
        x_iter, iter_count = solve_iterative(A, b, eps)
        print(f"Метод простых итераций:")
        print(f"Решение: {[round(x, 6) for x in x_iter]}")
        print(f"Итераций: {iter_count}")
        print(f"Проверка: A@x = {[round(x, 6) for x in matrix_vector_multiply(A, x_iter)]}\n")
    except ValueError as e:
        print(f"Ошибка в методе простых итераций: {e}")

    # Решение методом Зейделя
    try:
        x_seidel, seidel_count = solve_seidel(A, b, eps)
        print(f"Метод Зейделя:")
        print(f"Решение: {[round(x, 6) for x in x_seidel]}")
        print(f"Итераций: {seidel_count}")
        print(f"Проверка: A@x = {[round(x, 6) for x in matrix_vector_multiply(A, x_seidel)]}")
    except ValueError as e:
        print(f"Ошибка в методе Зейделя: {e}")