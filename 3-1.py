import math


def trigonometric_function(value):
    """Вычисляет тангенс значения"""
    return math.tan(value)


def build_lagrange_polynomial(x_coords, y_coords):
    """Строит строковое представление полинома Лагранжа"""
    polynomial = 'L(x) ='

    for i in range(len(x_coords)):
        numerator = ''
        denominator = 1

        for j in range(len(x_coords)):
            if i == j:
                continue
            numerator += f'(x-{x_coords[j]:.4f})'
            denominator *= (x_coords[i] - x_coords[j])

        if i > 0:
            polynomial += ' + '
        polynomial += f'{(y_coords[i] / denominator):.4f}*{numerator}'

    return polynomial


def calculate_lagrange_value(x_coords, y_coords, x_eval):
    """Вычисляет значение полинома Лагранжа в точке x_eval"""
    value = 0

    for i in range(len(x_coords)):
        term = y_coords[i] #/ (x_eval - x_coords[i])
        for j in range(len(x_coords)):
            if i != j:
                term *= (x_eval - x_coords[j]) / (x_coords[i] - x_coords[j])
        value += term

    return value


def compute_newton_coefficients(x_coords, y_coords):
    """Вычисляет коэффициенты для полинома Ньютона"""
    n = len(x_coords)
    coefficients = y_coords.copy()

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefficients[j] = (coefficients[j] - coefficients[j - 1]) / (x_coords[j] - x_coords[j - i])

    return coefficients


def build_newton_polynomial(x_coords, coefficients):
    """Строит строковое представление полинома Ньютона"""
    polynomial = 'P(x) = '
    term = ''

    for i in range(len(coefficients)):
        if i == 0:
            polynomial += f'{coefficients[i]:.4f}'
        else:
            polynomial += ' + ' + term + '*' + f'{coefficients[i]:.4f}'

        if i < len(coefficients) - 1:
            term += f'(x-{x_coords[i]:.4f})'

    return polynomial


def calculate_newton_value(x_coords, coefficients, x_eval):
    """Вычисляет значение полинома Ньютона в точке x_eval"""
    value = coefficients[0]
    product = 1

    for i in range(1, len(coefficients)):
        product *= (x_eval - x_coords[i - 1])
        value += coefficients[i] * product

    return value


def main():
    point_sets = [
        [0, math.pi / 8, math.pi / 4, 3 * math.pi / 8],
        [0, math.pi / 8, math.pi / 3, 3 * math.pi / 8]
    ]

    test_x = 3 * math.pi / 16
    test_y = trigonometric_function(test_x)

    print("Интерполяция функции tan(x)")
    print("=" * 50)

    for set_num, x_coords in enumerate(point_sets, 1):
        y_coords = [trigonometric_function(x) for x in x_coords]

        print(f"\nНабор точек #{set_num}:")
        print("X:", " ".join(f"{x:.4f}" for x in x_coords))
        print("Y:", " ".join(f"{y:.4f}" for y in y_coords))

        # Метод Лагранжа
        lagrange_poly = build_lagrange_polynomial(x_coords, y_coords)
        lagrange_value = calculate_lagrange_value(x_coords, y_coords, test_x)
        lagrange_error = abs(lagrange_value - test_y)

        print("\nМетод Лагранжа:")
        print(lagrange_poly)
        print(f"Значение в точке: {lagrange_value:.6f}")
        print(f"Ошибка: {lagrange_error:.6f}")

        # Метод Ньютона
        newton_coeffs = compute_newton_coefficients(x_coords, y_coords)
        newton_poly = build_newton_polynomial(x_coords, newton_coeffs)
        newton_value = calculate_newton_value(x_coords, newton_coeffs, test_x)
        newton_error = abs(newton_value - test_y)

        print("\nМетод Ньютона:")
        print(newton_poly)
        print(f"Значение в точке: {newton_value:.6f}")
        print(f"Ошибка: {newton_error:.6f}")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()