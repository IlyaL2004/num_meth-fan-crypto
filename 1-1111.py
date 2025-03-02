import numpy as np

def vector_inf_norm(v):
    """Вычисляет L∞-норму для вектора (максимальный по модулю элемент)"""
    return np.max(np.abs(v))

def matrix_inf_norm(M):
    """Вычисляет L∞-норму для матрицы (максимальная сумма модулей элементов строки)"""
    max_sum = 0
    # Перебираем все строки матрицы
    for row in M:
        row_sum = 0
        # Суммируем абсолютные значения элементов строки
        for element in row:
            row_sum += abs(element)
        # Обновляем максимальную сумму
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum


def solve_iterative(A, b, eps):
    """
    Решает СЛАУ Ax = b методом простых итераций
    Возвращает решение и количество итераций
    """
    n = A.shape[0]

    # Проверка диагональных элементов
    if np.any(np.diag(A) == 0):
        raise ValueError("Матрица содержит нулевые диагональные элементы")

    # Преобразование к виду x = alpha*x + beta
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')

    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    # Итерационный процесс
    cur_x = beta.copy()
    alpha_norm = matrix_inf_norm(alpha)
    iteration = 0

    while True:
        prev_x = cur_x.copy()

        # Ручное матрично-векторное умножение
        new_x = np.zeros_like(cur_x)
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                sum_val += alpha[i, j] * prev_x[j]
            new_x[i] = sum_val + beta[i]

        cur_x = new_x
        delta = vector_inf_norm(cur_x - prev_x)
        iteration += 1

        # Условие остановки
        if alpha_norm < 1:
            error_estimate = (alpha_norm / (1 - alpha_norm)) * delta
        else:
            error_estimate = delta

        if error_estimate <= eps:
            return cur_x, iteration


def solve_seidel(A, b, eps):
    """
    Решает СЛАУ Ax = b методом Зейделя
    Возвращает решение и количество итераций
    """
    n = A.shape[0]

    # Проверка диагональных элементов
    if np.any(np.diag(A) == 0):
        raise ValueError("Матрица содержит нулевые диагональные элементы")

    # Преобразование к виду x = alpha*x + beta
    alpha = np.zeros_like(A, dtype='float')
    beta = np.zeros_like(b, dtype='float')

    for i in range(n):
        beta[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                alpha[i, j] = -A[i, j] / A[i, i]

    # Итерационный процесс
    cur_x = beta.copy()
    alpha_norm = matrix_inf_norm(alpha)
    iteration = 0

    while True:
        prev_x = cur_x.copy()

        # Обновление по методу Зейделя
        for i in range(n):
            sum_val = 0.0
            # Используем уже обновленные значения для j < i
            for j in range(i):
                sum_val += alpha[i, j] * cur_x[j]
            # Используем старые значения для j >= i
            for j in range(i, n):
                sum_val += alpha[i, j] * prev_x[j]
            cur_x[i] = sum_val + beta[i]

        delta = vector_inf_norm(cur_x - prev_x)
        iteration += 1

        # Условие остановки
        if alpha_norm < 1:
            error_estimate = (alpha_norm / (1 - alpha_norm)) * delta
        else:
            error_estimate = delta

        if error_estimate <= eps:
            return cur_x, iteration





if __name__ == "__main__":
    # Пример хорошо обусловленной системы
    A = np.array([[4, 1],
                  [1, 3]], dtype='float')
    b = np.array([7, 10], dtype='float')

    # Пример плохо обусловленной системы (для демонстрации предупреждения)
    # A = np.array([[1, 2],
    #              [1, 1.999]], dtype='float')
    # b = np.array([3, 2.999], dtype='float')

    try:
        print("Исходная система:")
        print("Матрица A:\n", A)
        print("Вектор b:", b)
        print("\nЗапуск метода простых итераций...")

        solution, iterations = solve_iterative(A, b, 1e-6)

        print("\nРезультаты:")
        print(f"Приближенное решение: {np.round(solution, 6)}")
        print(f"Количество итераций: {iterations}")
        print("Проверка: A @ solution =", np.round(A @ solution, 6))

    except ValueError as e:
        print("\nОшибка:", e)

    try:
        # ... (предыдущий вывод)

        # Добавляем вызов метода Зейделя
        print("\nЗапуск метода Зейделя...")
        seidel_solution, seidel_iter = solve_seidel(A, b, 1e-6)

        print("\nРезультаты метода Зейделя:")
        print(f"Приближенное решение: {np.round(seidel_solution, 6)}")
        print(f"Количество итераций: {seidel_iter}")
        print("Проверка: A @ solution =", np.round(A @ seidel_solution, 6))

    except ValueError as e:
        print("\nОшибка:", e)