def algorithm(c, b, a, d):
    n = len(d)
    if n == 0:
        return []

    if n > 1:
        if len(c) < n - 1:
            raise ValueError(f"Длина массива c ({len(c)}) меньше требуемой ({n - 1})")
        for i in range(n - 1):
            if c[i] == 0:
                raise ValueError(f"Верхнедиагональный элемент c[{i}] равен нулю")

    alpha = [0.0] * n
    beta = [0.0] * n
    x = [0.0] * n
    alpha[0] = -c[0] / b[0] if n > 1 else 0  # Для n=1 не используется
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = a[i] * alpha[i - 1] + b[i]
        if i < n - 1:
            alpha[i] = -c[i] / denominator
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denominator

    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]


    print(f"P: {alpha}")
    print(f"Q: {beta}")

    return x


if __name__ == "__main__":
    c = [-5.0, -5.0, -6.0, -5.0, 0.0]  # Верхняя диагональ (последний элемент игнорируется)
    b = [13.0, 9.0, -12.0, 20.0, 5.0]  # Главная диагональ
    a = [0.0, -4.0, -1.0, 6.0, 4.0]  # Нижняя диагональ
    d = [-66.0, -47.0, -43.0, -74.0, 14.0]

    solution = algorithm(c, b, a, d)
    print("Решение:", [round(x, 3) for x in solution])