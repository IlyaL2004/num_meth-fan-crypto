import math


def integrand(x):
    """Математическая функция для интегрирования"""
    return x / (3 * x + 4) ** 3


def midpoint_integration(func, a, b, step_size):
    """
    Численное интегрирование методом средних прямоугольников

    Args:
        func (callable): Интегрируемая функция
        a (float): Левая граница интервала
        b (float): Правая граница интервала
        step_size (float): Шаг интегрирования

    Returns:
        float: Приближённое значение интеграла
    """
    total = 0.0
    current = a
    while current < b:
        midpoint = current + step_size / 2
        if midpoint > b:
            midpoint = (current + b) / 2
        total += func(midpoint) * min(step_size, b - current)
        current += step_size
    return total


def trapezoidal_rule(func, a, b, step_size):
    """
    Численное интегрирование по составной формуле трапеций

    Args:
        func (callable): Интегрируемая функция
        a (float): Левая граница интервала
        b (float): Правая граница интервала
        step_size (float): Шаг интегрирования

    Returns:
        float: Приближённое значение интеграла
    """
    integration_sum = 0.0
    x_left = a
    while x_left < b - 1e-9:
        x_right = min(x_left + step_size, b)
        integration_sum += 0.5 * (func(x_left) + func(x_right)) * (x_right - x_left)
        x_left = x_right
    return integration_sum


def simpsons_integrator(func, a, b, step_size):
    """
    Численное интегрирование по составному правилу Симпсона

    Args:
        func (callable): Интегрируемая функция
        a (float): Левая граница интервала
        b (float): Правая граница интервала
        step_size (float): Шаг интегрирования

    Returns:
        float: Приближённое значение интеграла
    """

    integral = func(a) + func(b)
    num_points = int((b - a) // (2 * step_size))

    for i in range(1, num_points + 1):
        x = a + (2 * i - 1) * step_size
        integral += 4 * func(x)

    for i in range(1, num_points):
        x = a + 2 * i * step_size
        integral += 2 * func(x)

    return integral * step_size / 3


def richardson_extrapolation(step_coarse, step_fine, result_coarse, result_fine, order):
    """
    Уточнение результата по Ричардсону

    Args:
        step_coarse (float): Большой шаг
        step_fine (float): Малый шаг
        result_coarse (float): Результат с большим шагом
        result_fine (float): Результат с малым шагом
        order (int): Порядок метода

    Returns:
        float: Уточнённое значение интеграла
    """
    ratio = step_fine / step_coarse
    return result_fine + (result_fine - result_coarse) / (ratio ** -order - 1)


if __name__ == "__main__":
    lower_bound = -1.0
    upper_bound = 1.0
    coarse_step = 0.5
    fine_step = 0.25

    methods = {
        "Метод средних прямоугольников": midpoint_integration,
        "Составная трапеция": trapezoidal_rule,
        "Правило Симпсона": simpsons_integrator
    }

    results = {}
    for method_name, integrator in methods.items():
        coarse_result = integrator(integrand, lower_bound, upper_bound, coarse_step)
        fine_result = integrator(integrand, lower_bound, upper_bound, fine_step)
        results[method_name] = (coarse_result, fine_result)
        print(f"{method_name}:")
        print(f"  Шаг {coarse_step}: {coarse_result:.5f}")
        print(f"  Шаг {fine_step}: {fine_result:.5f}")

    print("\nЭкстраполяция Ричардсона:")
    orders = {"Метод средних прямоугольников": 2,
              "Составная трапеция": 2,
              "Правило Симпсона": 4}

    for method_name, (coarse, fine) in results.items():
        refined = richardson_extrapolation(coarse_step, fine_step, coarse, fine, orders[method_name])
        print(f"{method_name}: {refined:.6f}")