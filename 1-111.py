
def thomas_algorithm(c, b, a, d):  # Изменен порядок аргументов: c (верхняя), b (главная), a (нижняя)
    n = len(d)
    if n == 0:
        return []

    alpha = [0.0] * n
    beta = [0.0] * n
    x = [0.0] * n
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = a[i] * alpha[i - 1] + b[i]
        if i < n - 1:
            alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

    # Обратный ход
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


if __name__ == "__main__":
    # Данные для системы из 5 уравнений (теперь порядок: c, b, a)
    c = [-5.0, -5.0, -6.0, -5.0, 0.0]  # Верхняя диагональ (c[-1] не используется)
    b = [13.0, 9.0, -12.0, 20.0, 5.0]  # Главная диагональ
    a = [0.0, -4.0, -1.0, 6.0, 4.0]  # Нижняя диагональ (a[0] не используется)
    d = [-66.0, -47.0, -43.0, -74.0, 14.0]  # Вектор правых частей

    solution = thomas_algorithm(c, b, a, d)  # Передаем данные в новом порядке
    print("Решение:", [round(x, 3) for x in solution])