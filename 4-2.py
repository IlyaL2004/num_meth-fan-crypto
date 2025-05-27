import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def exact_solution(x):
    return 1 / x + 1


def ode_system(x, y):
    return [y[1], 2 * y[0] / (x ** 2 * (x + 1))]


def shooting_method(a, b, h, max_iter=100, tol=1e-6):

    def residual(eta, step):
        t_eval = np.linspace(a, b, int((b - a) / step) + 1)
        sol = solve_ivp(
            ode_system, [a, b], [eta, -1],
            method='RK45',
            t_eval=t_eval,
            max_step=step,
            rtol=1e-6,
            atol=1e-9
        )
        yb, ypb = sol.y[0, -1], sol.y[1, -1]
        return 2 * yb - 4 * ypb - 4

    eta0, eta1 = 1.5, 2.0
    f0 = residual(eta0, h)
    f1 = residual(eta1, h)
    for _ in range(max_iter):
        if abs(f1) < tol:
            break
        denom = f1 - f0
        if abs(denom) < 1e-14:
            break
        eta_new = eta1 - f1 * (eta1 - eta0) / denom
        eta0, f0 = eta1, f1
        eta1, f1 = eta_new, residual(eta_new, h)

    t_eval_h = np.linspace(a, b, int((b - a) / h) + 1)
    sol_h = solve_ivp(
        ode_system, [a, b], [eta1, -1],
        method='RK45',
        t_eval=t_eval_h,
        max_step=h,
        rtol=1e-6,
        atol=1e-9
    )

    h2 = h / 2
    t_eval_h2 = np.linspace(a, b, int((b - a) / h2) + 1)
    sol_h2 = solve_ivp(
        ode_system, [a, b], [eta1, -1],
        method='RK45',
        t_eval=t_eval_h2,
        max_step=h2,
        rtol=1e-6,
        atol=1e-9
    )

    return (sol_h.t, sol_h.y[0]), (sol_h2.t, sol_h2.y[0])


def finite_difference(a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)

    A = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)

    A[0, 0] = -1 / h
    A[0, 1] = 1 / h
    rhs[0] = -1

    for i in range(1, N):
        xi = x[i]
        A[i, i - 1] = xi ** 2 * (xi + 1)
        A[i, i] = -2 * xi ** 2 * (xi + 1) - 2 * h ** 2
        A[i, i + 1] = xi ** 2 * (xi + 1)

    A[N, N - 1] = 4 / h
    A[N, N] = 2 - 4 / h
    rhs[N] = 4

    y = np.linalg.solve(A, rhs)
    return x, y


def runge_romberg(y_coarse, y_fine, p):
    return np.max(np.abs(y_coarse - y_fine[::2])) / (2 ** p - 1)


def mae(y_num, y_exact):
    return np.mean(np.abs(y_num - y_exact))


if __name__ == "__main__":
    a, b = 1, 2
    N = 1000
    h = (b - a) / N

    (x_shoot, y_shoot), (x_shoot_h2, y_shoot_h2) = shooting_method(a, b, h)

    x_fd, y_fd       = finite_difference(a, b, N)
    x_fd_h2, y_fd_h2 = finite_difference(a, b, 2 * N)

    y_exact_shoot    = exact_solution(x_shoot)
    y_exact_shoot_h2 = exact_solution(x_shoot_h2)

    print("\nРезультаты:")
    print("=" * 65)
    print(f"{'Метод':<20} | {'Шаг':<10} | {'MAE':<15} | {'Рунге-Ромберг':<15}")
    print("-" * 65)
    print(f"{'Стрельба':<20} | {h:.4f}    | {mae(y_shoot, y_exact_shoot):.4e}    | {'-':<15}")
    print(
        f"{'Стрельба':<20} | {h / 2:.4f}    | {mae(y_shoot_h2, y_exact_shoot_h2):.4e}    | {runge_romberg(y_shoot, y_shoot_h2, 4):.4e}"
    )
    print(f"{'Конечно-разностный':<20} | {h:.4f}    | {mae(y_fd, exact_solution(x_fd)):.4e}    | {'-':<15}")
    print(
        f"{'Конечно-разностный':<20} | {h / 2:.4f}    | {mae(y_fd_h2, exact_solution(x_fd_h2)):.4e}    | {runge_romberg(y_fd, y_fd_h2, 2):.4e}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(x_shoot, y_shoot, 'r--', linewidth=3, label='Стрельба (h)')
    ax1.plot(x_fd,    y_fd,    'b-',  alpha=0.7, label='Конечно-разностный (h)')
    ax1.plot(x_shoot, y_exact_shoot, 'k-', label='Точное')
    ax1.set_title(f'h = {h:.4f}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y(x)')
    ax1.legend(); ax1.grid(True)

    ax2.plot(x_shoot_h2, y_shoot_h2, 'g--', linewidth=3, label='Стрельба (h/2)')
    ax2.plot(x_fd_h2,    y_fd_h2,    'm-',  alpha=0.7, label='Конечно-разностный (h/2)')
    ax2.plot(x_shoot_h2, y_exact_shoot_h2, 'k-', label='Точное')
    ax2.set_title(f'h/2 = {h/2:.4f}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y(x)')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()
