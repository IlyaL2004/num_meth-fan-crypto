import numpy as np
import matplotlib.pyplot as plt


def compute_cubic(coeff_a, coeff_b, coeff_c, coeff_d, x_val):
    """Вычисляет значение кубического сплайна"""
    return coeff_a + coeff_b * x_val + coeff_c * x_val ** 2 + coeff_d * x_val ** 3


def get_tridiag_components(matrix):
    """Извлекает компоненты трёхдиагональной матрицы"""
    size = len(matrix)
    upper_diag = []
    main_diag = []
    lower_diag = []

    for row_idx in range(size):
        for col_idx in range(size):
            if col_idx == row_idx + 1:
                upper_diag.append(matrix[row_idx][col_idx])
            elif col_idx == row_idx:
                main_diag.append(matrix[row_idx][col_idx])
            elif col_idx == row_idx - 1:
                lower_diag.append(matrix[row_idx][col_idx])
    return upper_diag, main_diag, lower_diag


def solve_tridiagonal(upper, main, lower, rhs):
    """Решает трёхдиагональную систему методом прогонки"""
    system_size = len(rhs)
    if system_size == 0:
        return []

    if system_size > 1:
        if len(upper) < system_size - 1:
            raise ValueError("Несовпадение размеров верхней диагонали")
        for i in range(system_size - 1):
            if upper[i] == 0:
                raise ValueError("Нулевой элемент в верхней диагонали")

    alpha_coeffs = [0.0] * system_size
    beta_coeffs = [0.0] * system_size
    solution = [0.0] * system_size

    if system_size > 1:
        alpha_coeffs[0] = -upper[0] / main[0]
    beta_coeffs[0] = rhs[0] / main[0]

    for idx in range(1, system_size):
        divisor = lower[idx - 1] * alpha_coeffs[idx - 1] + main[idx]
        if idx < system_size - 1:
            alpha_coeffs[idx] = -upper[idx] / divisor
        beta_coeffs[idx] = (rhs[idx] - lower[idx - 1] * beta_coeffs[idx - 1]) / divisor

    solution[-1] = beta_coeffs[-1]
    for idx in range(system_size - 2, -1, -1):
        solution[idx] = alpha_coeffs[idx] * solution[idx + 1] + beta_coeffs[idx]

    print("Коэффициенты альфа:", alpha_coeffs)
    print("Коэффициенты бета:", beta_coeffs)

    return solution


def build_cubic_spline(nodes_x, nodes_y, test_point):
    """Строит кубический сплайн и вычисляет значение в заданной точке"""
    num_points = len(nodes_x)


    step_sizes = [nodes_x[i] - nodes_x[i - 1] for i in range(1, len(nodes_x))]


    matrix_size = len(step_sizes) - 1
    tridiag_matrix = [[0] * matrix_size for _ in range(matrix_size)]
    tridiag_matrix[0][0] = 2 * (step_sizes[0] + step_sizes[1])
    tridiag_matrix[0][1] = step_sizes[1]

    for i in range(1, matrix_size - 1):
        tridiag_matrix[i][i - 1] = step_sizes[i]
        tridiag_matrix[i][i] = 2 * (step_sizes[i] + step_sizes[i + 1])
        tridiag_matrix[i][i + 1] = step_sizes[i + 1]

    tridiag_matrix[-1][-2] = step_sizes[-2]
    tridiag_matrix[-1][-1] = 2 * (step_sizes[-2] + step_sizes[-1])

    rhs_vector = [3.0 * ((nodes_y[i + 1] - nodes_y[i]) / step_sizes[i] -
                         (nodes_y[i] - nodes_y[i - 1]) / step_sizes[i - 1])
                  for i in range(1, len(step_sizes))]

    upper, main, lower = get_tridiag_components(tridiag_matrix)
    c_coeffs = [0] + solve_tridiagonal(upper, main, lower, rhs_vector)


    a_coeffs = [nodes_y[i - 1] for i in range(1, num_points)]


    b_coeffs = [
    (nodes_y[i] - nodes_y[i - 1]) / step_sizes[i - 1] -
    (step_sizes[i - 1] / 3.0) * (2 * c_coeffs[i - 1] + c_coeffs[i])
    for i in range(1, len(step_sizes))]
    b_coeffs.append(
    (nodes_y[-1] - nodes_y[-2]) / step_sizes[-1] -
    (2 * step_sizes[-1] * c_coeffs[-1]) / 3.0
    )

    d_coeffs = [
    (c_coeffs[i] - c_coeffs[i - 1]) / (3.0 * step_sizes[i - 1])
    for i in range(1, len(step_sizes))
        ]
    d_coeffs.append(-c_coeffs[-1] / (3.0 * step_sizes[-1]))


    interval_idx = next(
    i
    for i in range(len(nodes_x) - 1)
        if nodes_x[i] <= test_point < nodes_x[i + 1]
    )

    return (
        a_coeffs,
        b_coeffs,
        c_coeffs,
        d_coeffs,
        compute_cubic(
            a_coeffs[interval_idx + 1],
            b_coeffs[interval_idx + 1],
            c_coeffs[interval_idx + 1],
            d_coeffs[interval_idx + 1],
            test_point - nodes_x[interval_idx]
        )
    )


def visualize_spline(original_x, original_y, a, b, c, d):
    """Визуализирует сплайн и исходные точки"""
    spline_x, spline_y = [], []
    for i in range(len(original_x) - 1):
        x_vals = np.linspace(original_x[i], original_x[i + 1], 10)
        y_vals = [compute_cubic(a[i], b[i], c[i], d[i], x - original_x[i]) for x in x_vals]
        spline_x.append(x_vals)
        spline_y.append(y_vals)

    plt.scatter(original_x, original_y, color='red', label='Исходные точки')
    for i in range(len(original_x) - 1):
        plt.plot(spline_x[i], spline_y[i], color='blue', label='Сплайн' if i == 0 else "")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    input_x = [0.0, 0.9, 1.8, 2.7, 3.6]
    input_y = [0.0, 0.36892, 0.85408, 1.7856, 6.3138]
    test_x = 1.5

    a_coeffs, b_coeffs, c_coeffs, d_coeffs, result = build_cubic_spline(input_x, input_y, test_x)

    for i in range(len(input_x) - 1):
        print(f'Интервал [{input_x[i]}; {input_x[i + 1]}):')
        print(
            f's(x) = {a_coeffs[i]} + {b_coeffs[i]}(x-{input_x[i]}) + {c_coeffs[i]}(x-{input_x[i]})² + {d_coeffs[i]}(x-{input_x[i]})³')

    print(f'Результат: s({test_x}) = {result:.4f}')
    visualize_spline(input_x, input_y, a_coeffs, b_coeffs, c_coeffs, d_coeffs)
    print("\nКоэффициенты кубического сплайна:")
    print("--------------------------------")
    print(f"{'Интервал':<15} {'a':<15} {'b':<15} {'c':<15} {'d':<15}")
    for i in range(len(input_x) - 1):
        print(f"[{input_x[i]:.1f}, {input_x[i + 1]:.1f}): "
              f"{a_coeffs[i]:<15.6f} {b_coeffs[i]:<15.6f} "
              f"{c_coeffs[i]:<15.6f} {d_coeffs[i]:<15.6f}")

    print("\nЗначение сплайна в точке x =", test_x, ":", f"{result:.6f}")