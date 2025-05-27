import matplotlib.pyplot as plt


def factorize_matrix_lu(matrix):
    size = len(matrix)
    upper = [row.copy() for row in matrix]
    lower = [[0.0] * size for _ in range(size)]
    permutations = list(range(size))
    swaps = 0

    for col in range(size):
        pivot = col
        max_value = abs(upper[col][col])
        for row in range(col, size):
            if abs(upper[row][col]) > max_value:
                pivot = row
                max_value = abs(upper[row][col])

        if pivot != col:
            upper[col], upper[pivot] = upper[pivot], upper[col]
            for j in range(col):
                lower[col][j], lower[pivot][j] = lower[pivot][j], lower[col][j]
            permutations[col], permutations[pivot] = permutations[pivot], permutations[col]
            swaps += 1

        if upper[col][col] == 0:
            raise ArithmeticError("Матрица необратима")

        for row in range(col + 1, size):
            factor = upper[row][col] / upper[col][col]
            lower[row][col] = factor
            for j in range(col, size):
                upper[row][j] -= factor * upper[col][j]

        for diag in range(size):
            lower[diag][diag] = 1.0

    return permutations, lower, upper, swaps


def compute_lower_part(lower, vector):
    n = len(lower)
    intermediate = [0.0] * n
    for i in range(n):
        accum = 0.0
        for j in range(i):
            accum += lower[i][j] * intermediate[j]
        intermediate[i] = vector[i] - accum
    return intermediate


def compute_upper_part(upper, intermediate):
    n = len(upper)
    solution = [0.0] * n
    for i in reversed(range(n)):
        accum = 0.0
        for j in range(i + 1, n):
            accum += upper[i][j] * solution[j]
        solution[i] = (intermediate[i] - accum) / upper[i][i]
    return solution


def solve_linear_system(matrix, rhs, perm, lower, upper):
    reordered_rhs = [rhs[perm[i]] for i in range(len(perm))]
    intermediate = compute_lower_part(lower, reordered_rhs)
    solution = compute_upper_part(upper, intermediate)
    return solution


def compute_regression_model(inputs, outputs, degree):
    basis_size = degree + 1
    coefficients_matrix = []
    constants_vector = []

    for row in range(basis_size):
        matrix_row = []
        for col in range(basis_size):
            total = sum(x ** (row + col) for x in inputs)
            matrix_row.append(total)
        coefficients_matrix.append(matrix_row)

        vector_element = sum(y * (x ** row) for x, y in zip(inputs, outputs))
        constants_vector.append(vector_element)

    perm, lower, upper, _ = factorize_matrix_lu(coefficients_matrix)
    return solve_linear_system(coefficients_matrix, constants_vector, perm, lower, upper)


def evaluate_polynomial(coefficients, value):
    return sum(coeff * (value ** power) for power, coeff in enumerate(coefficients))


def calculate_residuals(inputs, outputs, coefficients):
    approximations = [evaluate_polynomial(coefficients, x) for x in inputs]
    return sum((actual - approx) ** 2 for actual, approx in zip(outputs, approximations))


def main():
    sample_x = [-0.9, 0.0, 0.9, 1.8, 2.7, 3.6]
    sample_y = [-0.36892, 0.0, 0.36892, 0.85408, 1.7856, 6.3138]

    plt.figure(figsize=(10, 6))
    plt.scatter(sample_x, sample_y, c='red', marker='o', label='Исходные точки')
    plt.plot(sample_x, sample_y, 'c--', alpha=0.5, label='Исходные данные')

    # Линейная аппроксимация
    lin_coeffs = compute_regression_model(sample_x, sample_y, 1)
    print(f"Линейная модель: y = {lin_coeffs[0]:.4f} + {lin_coeffs[1]:.4f}x")
    plt.plot(sample_x, [evaluate_polynomial(lin_coeffs, x) for x in sample_x],
             'b', label='Линейная аппроксимация')

    # Квадратичная аппроксимация
    quad_coeffs = compute_regression_model(sample_x, sample_y, 2)
    print(f"Квадратичная модель: y = {quad_coeffs[0]:.4f} + {quad_coeffs[1]:.4f}x + {quad_coeffs[2]:.4f}x²")
    plt.plot(sample_x, [evaluate_polynomial(quad_coeffs, x) for x in sample_x],
             'g', label='Квадратичная аппроксимация')

    print(f"\nОшибки:")
    print(f"Линейная модель: {calculate_residuals(sample_x, sample_y, lin_coeffs):.6f}")
    print(f"Квадратичная модель: {calculate_residuals(sample_x, sample_y, quad_coeffs):.6f}")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Аппроксимация данных')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()