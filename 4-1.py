import numpy as np
import matplotlib.pyplot as plt


def derivative_func(x, current_y, derivative):
    return 2 * current_y + 4 * x ** 2 * np.exp(x ** 2)


def state_func(x, current_y, derivative):
    return derivative


def exact_solution(x):
    return np.exp(x ** 2) + np.exp(x * np.sqrt(2)) + np.exp(-x * np.sqrt(2))


def euler_method(deriv_func, state_func, initial_y, initial_deriv, interval, step_size):
    start, end = interval
    x_points = np.arange(start, end + step_size, step_size)
    y_points = [initial_y]
    current_deriv = initial_deriv

    for i in range(len(x_points) - 1):
        current_deriv += step_size * deriv_func(x_points[i], y_points[i], current_deriv)
        y_points.append(y_points[i] + step_size * state_func(x_points[i], y_points[i], current_deriv))

    return x_points, y_points


def runge_kutta_method(deriv_func, state_func, initial_y, initial_deriv, interval, step_size, return_deriv=False):
    start, end = interval
    x_points = list(np.arange(start, end + step_size, step_size))
    y_points = [initial_y]
    deriv_points = [initial_deriv]

    for i in range(len(x_points) - 1):
        k1 = step_size * state_func(x_points[i], y_points[i], deriv_points[i])
        l1 = step_size * deriv_func(x_points[i], y_points[i], deriv_points[i])

        k2 = step_size * state_func(x_points[i] + 0.5 * step_size,
                                    y_points[i] + 0.5 * k1,
                                    deriv_points[i] + 0.5 * l1)
        l2 = step_size * deriv_func(x_points[i] + 0.5 * step_size,
                                    y_points[i] + 0.5 * k1,
                                    deriv_points[i] + 0.5 * l1)

        k3 = step_size * state_func(x_points[i] + 0.5 * step_size,
                                    y_points[i] + 0.5 * k2,
                                    deriv_points[i] + 0.5 * l2)
        l3 = step_size * deriv_func(x_points[i] + 0.5 * step_size,
                                    y_points[i] + 0.5 * k2,
                                    deriv_points[i] + 0.5 * l2)

        k4 = step_size * state_func(x_points[i] + step_size,
                                    y_points[i] + k3,
                                    deriv_points[i] + l3)
        l4 = step_size * deriv_func(x_points[i] + step_size,
                                    y_points[i] + k3,
                                    deriv_points[i] + l3)

        y_update = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        deriv_update = (l1 + 2 * l2 + 2 * l3 + l4) / 6

        y_points.append(y_points[i] + y_update)
        deriv_points.append(deriv_points[i] + deriv_update)

    return (x_points, y_points, deriv_points) if return_deriv else (x_points, y_points)


def adams_method(deriv_func, state_func, initial_y, initial_deriv, interval, step_size):
    rk_x, rk_y, rk_deriv = runge_kutta_method(deriv_func, state_func, initial_y, initial_deriv, interval, step_size, True)

    x_points = rk_x
    y_points = rk_y[:4]
    deriv_points = rk_deriv[:4]

    for i in range(3, len(x_points) - 1):
        new_deriv = deriv_points[i] + step_size * (
                55 * deriv_func(x_points[i], y_points[i], deriv_points[i])
                - 59 * deriv_func(x_points[i - 1], y_points[i - 1], deriv_points[i - 1])
                + 37 * deriv_func(x_points[i - 2], y_points[i - 2], deriv_points[i - 2])
                - 9 * deriv_func(x_points[i - 3], y_points[i - 3], deriv_points[i - 3])
        ) / 24

        new_y = y_points[i] + step_size * (
                55 * state_func(x_points[i], y_points[i], deriv_points[i])
                - 59 * state_func(x_points[i - 1], y_points[i - 1], deriv_points[i - 1])
                + 37 * state_func(x_points[i - 2], y_points[i - 2], deriv_points[i - 2])
                - 9 * state_func(x_points[i - 3], y_points[i - 3], deriv_points[i - 3])
        ) / 24

        deriv_points.append(new_deriv)
        y_points.append(new_y)

    return x_points, y_points


def runge_romberg_error(approx_h, approx_h2, order):
    return np.mean(np.abs(np.array(approx_h) - np.array(approx_h2[::2]))) / (2 ** order - 1)


def mean_absolute_error(approx, exact):
    return np.mean(np.abs(np.array(approx) - np.array(exact)))


if __name__ == '__main__':
    INITIAL_Y = 3
    INITIAL_DERIV = 0
    INTERVAL = (0, 1)
    BASE_STEP = 0.1

    # Вычисления с базовым шагом
    euler_x, euler_y = euler_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, BASE_STEP)
    rk_x, rk_y = runge_kutta_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, BASE_STEP)
    adams_x, adams_y = adams_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, BASE_STEP)

    # Вычисления с половинным шагом
    half_step = BASE_STEP / 2
    euler_x_half, euler_y_half = euler_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, half_step)
    rk_x_half, rk_y_half = runge_kutta_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, half_step)
    adams_x_half, adams_y_half = adams_method(derivative_func, state_func, INITIAL_Y, INITIAL_DERIV, INTERVAL, half_step)

    # Точное решение
    exact_x = np.arange(INTERVAL[0], INTERVAL[1] + BASE_STEP, BASE_STEP)
    exact_y = [exact_solution(x) for x in exact_x]
    exact_x_half = np.arange(INTERVAL[0], INTERVAL[1] + half_step, half_step)
    exact_y_half = [exact_solution(x) for x in exact_x_half]

    # Создание трех графиков
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # График для метода Эйлера
    axes[0].plot(euler_x, euler_y, '--', color='blue', label=f'h = {BASE_STEP}')
    axes[0].plot(euler_x_half, euler_y_half, '-.', color='red', label=f'h = {half_step}')
    axes[0].plot(exact_x, exact_y, 'k-', label='Exact Solution')
    axes[0].set_title('Euler Method Comparison')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True)
    axes[0].legend()

    # График для метода Рунге-Кутты
    axes[1].plot(rk_x, rk_y, '--', color='green', label=f'h = {BASE_STEP}')
    axes[1].plot(rk_x_half, rk_y_half, '-.', color='orange', label=f'h = {half_step}')
    axes[1].plot(exact_x, exact_y, 'k-', label='Exact Solution')
    axes[1].set_title('Runge-Kutta Method Comparison')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True)
    axes[1].legend()

    # График для метода Адамса
    axes[2].plot(adams_x, adams_y, '--', color='purple', label=f'h = {BASE_STEP}')
    axes[2].plot(adams_x_half, adams_y_half, '-.', color='brown', label=f'h = {half_step}')
    axes[2].plot(exact_x, exact_y, 'k-', label='Exact Solution')
    axes[2].set_title('Adams Method Comparison')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Вывод ошибок
    print("Mean Absolute Errors:")
    print(f"Euler (h={BASE_STEP}): {mean_absolute_error(euler_y, exact_y):.2e}")
    print(f"Euler (h={half_step}): {mean_absolute_error(euler_y_half, exact_y_half):.2e}\n")

    print(f"Runge-Kutta (h={BASE_STEP}): {mean_absolute_error(rk_y, exact_y):.2e}")
    print(f"Runge-Kutta (h={half_step}): {mean_absolute_error(rk_y_half, exact_y_half):.2e}\n")

    print(f"Adams (h={BASE_STEP}): {mean_absolute_error(adams_y, exact_y):.2e}")
    print(f"Adams (h={half_step}): {mean_absolute_error(adams_y_half, exact_y_half):.2e}\n")

    print("Runge-Romberg Errors:")
    print(f"Euler: {runge_romberg_error(euler_y, euler_y_half, 1):.2e}")
    print(f"Runge-Kutta: {runge_romberg_error(rk_y, rk_y_half, 4):.2e}")
    print(f"Adams: {runge_romberg_error(adams_y, adams_y_half, 4):.2e}")