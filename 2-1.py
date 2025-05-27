

#Пусть на отрезке [a,b] функция имеет первую и вторую производные постоянного знака
import math
import numpy as np


def newton_method(f, f_prime, f_second, x0, a, b, epsilon=1e-6, max_iter=100):
    # Проверка условия f(a)*f(b) < 0
    if f(a) * f(b) >= 0:
        raise ValueError("Условие f(a)*f(b) < 0 не выполняется")

    # Проверка условия f(x0)*f''(x0) > 0
    if f(x0) * f_second(x0) <= 0:
        raise ValueError("Начальное приближение не удовлетворяет условию f(x0)*f''(x0) > 0")

    x_prev = x0
    for i in range(max_iter):
        try:
            x_next = x_prev - f(x_prev) / f_prime(x_prev)
        except ZeroDivisionError:
            raise ValueError("Производная равна нулю. Метод расходится.")

        # Проверка, что следующая точка остается в [a, b]
        if x_next < a or x_next > b:
            raise ValueError(f"Итерация вышла за пределы отрезка [a, b]: x = {x_next}")

        if abs(x_next - x_prev) < epsilon:
            return x_next, i + 1

        x_prev = x_next

    raise ValueError(f"Метод не сошёлся за {max_iter} итераций")


def simple_iteration(phi, phi_derivative, a, b, x0, epsilon=1e-6, max_iter=100, num_test_points=20):
    if x0 < a or x0 > b:
        raise ValueError("Начальное приближение вне отрезка [a, b]")

    # Проверка условия phi(x) ∈ [a, b] для всех x ∈ [a, b]
    test_points = np.linspace(a, b, num=num_test_points)
    for x in test_points:
        try:
            phi_x = phi(x)
        except ValueError:
            raise ValueError(f"Функция phi(x) не определена при x = {x:.3f}")
        if phi_x < a or phi_x > b:
            raise ValueError(f"Условие phi(x) ∈ [a, b] нарушено при x = {x:.3f}")

    # Проверка условия |phi'(x)| ≤ q < 1
    q_values = [abs(phi_derivative(x)) for x in test_points]
    q = max(q_values)
    if q >= 1:
        raise ValueError(f"Условие сходимости не выполняется (q = {q:.2f} >= 1)")

    x_prev = x0
    for i in range(max_iter):
        x_next = phi(x_prev)

        # Критерий остановки с учетом q
        if abs(x_next - x_prev) < epsilon * (1 - q) / q:
            return x_next, i + 1

        x_prev = x_next

    raise ValueError(f"Метод не сошёлся за {max_iter} итераций")


if __name__ == "__main__":
    def f(x):
        return math.sqrt(1 - x ** 2) - math.exp(x) + 0.1


    def f_prime(x):
        return (-x / math.sqrt(1 - x ** 2)) - math.exp(x)


    def f_second(x):
        return (-1 / (1 - x ** 2) ** 1.5) - math.exp(x)


    def phi(x):
        return math.log(math.sqrt(1 - x ** 2) + 0.1)


    def phi_derivative(x):
        denominator = math.sqrt(1 - x ** 2) * (math.sqrt(1 - x ** 2) + 0.1)
        return (-x) / denominator if denominator != 0 else float('inf')


    a, b = 0.00, 0.20
    x0 = 0.15

    print("Метод Ньютона:")
    try:
        root, iterations = newton_method(f, f_prime, f_second, x0, a, b)
        print(f"Найденный корень: {root:.8f}")
        print(f"Количество итераций: {iterations}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\nМетод простой итерации:")
    try:
        root, iterations = simple_iteration(phi, phi_derivative, a, b, x0, num_test_points=50)
        print(f"Найденный корень: {root:.8f}")
        print(f"Количество итераций: {iterations}")
        print(f"Максимальное q: {max(abs(phi_derivative(x)) for x in np.linspace(a, b, 50)):.4f}")
    except Exception as e:
        print(f"Ошибка: {e}")