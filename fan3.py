import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Параметры
k = 6
a = -0.8 - k / 10
b = 0.8 + k / 10

t = sp.symbols('t')
y = sp.sin(2 * t)

# Определение взвешенного скалярного произведения
def scalar(f1, f2):
    w = (4 - t)**2
    return sp.integrate(f1 * f2 * w, (t, a, b))

# Список точностей и структура для хранения результатов
#epsilons = [1e-1, 1e-2, 1e-3]
epsilons = [1e-1]

results = {}

for epsilon in epsilons:
    print(f"\n=== Аппроксимация с точностью ε = {epsilon} ===")

    basis = [sp.Integer(1)]
    c_list = []
    fourier_series = 0
    threshold_step = None
    errors = []

    phi_0 = basis[0]
    numer_0 = scalar(y, phi_0)
    denom_0 = scalar(phi_0, phi_0)
    c_0 = numer_0 / denom_0
    c_list.append(c_0)
    fourier_series += c_0 * phi_0

    print("φ_0(t) = 1")
    print("Общая формула: c_0 = ⟨y, φ_0⟩ / ⟨φ_0, φ_0⟩")
    print(f"  ⟨y, φ_0⟩ = {numer_0.evalf():.6f}")
    print(f"  ⟨φ_0, φ_0⟩ = {denom_0.evalf():.6f}")
    print(f"  c_0 = {c_0.evalf():.6f}")
    print(f"S_0(t) = {str(sp.simplify(fourier_series)).replace('**', '^')}")
    print("-" * 60)

    max_iter = 20
    for i in range(1, max_iter + 1):
        orig = t**i
        new_poly = orig
        formula = f"φ_{i}(t) = t^{i}"
        for j, phi_j in enumerate(basis):
            numer = scalar(orig, phi_j)
            denom = scalar(phi_j, phi_j)
            coeff = numer / denom
            formula += f" - ({coeff.evalf():.6f})·φ_{j}(t)"
            new_poly -= coeff * phi_j
            new_poly = sp.simplify(new_poly)
            print(f"  ⟨t^{i}, φ_{j}⟩ = {numer.evalf():.6f}")
            print(f"  ⟨φ_{j}, φ_{j}⟩ = {denom.evalf():.6f}")
            print(f"  c_{i}{j} = {coeff.evalf():.6f}")
        basis.append(new_poly)
        print(f"Общая формула: {formula}")
        print(f"Новый базисный вектор φ_{i}(t) = {str(new_poly).replace('**','^')}")

        numer_ci = scalar(y, new_poly)
        denom_ci = scalar(new_poly, new_poly)
        c_i = numer_ci / denom_ci
        c_list.append(c_i)
        fourier_series += c_i * new_poly
        print(f"Общая формула: c_{i} = ⟨y, φ_{i}⟩ / ⟨φ_{i}, φ_{i}⟩")
        print(f"  ⟨y, φ_{i}⟩ = {numer_ci.evalf():.6f}")
        print(f"  ⟨φ_{i}, φ_{i}⟩ = {denom_ci.evalf():.6f}")
        print(f"  c_{i} = {c_i.evalf():.6f}")

        partial_formula = "S_{i}(t) = " + " + ".join([f"c_{j}·φ_{j}(t)" for j in range(i+1)])
        print(f"Общая формула: {partial_formula}")
        for j in range(i+1):
            print(f"  Часть {j}: ({c_list[j].evalf():.6f})·({str(basis[j]).replace('**','^')})")
        simplified_Si = str(sp.simplify(fourier_series)).replace('**','^')
        print(f"Новая частичная сумма S_{i}(t) = {simplified_Si}")
        print("-" * 60)

        err_sq = scalar(y - fourier_series, y - fourier_series)
        err = sp.sqrt(err_sq).evalf()
        errors.append(float(err))
        print(f"Среднеквадратичная ошибка после {i} шагов: {err:.6f}")
        if threshold_step is None and err <= epsilon:
            threshold_step = i
            print(f"--> Достигнута точность ε={epsilon} при i={i}")
            break
    results[epsilon] = {'step': threshold_step, 'basis': basis, 'c_list': c_list}

print("\n=== Сводка по ε ===")
for eps, data in results.items():
    print(f"Для ε = {eps}: минимальный i = {data['step']}")

t_vals = np.linspace(a, b, 400)
y_vals = np.sin(2 * t_vals)
for epsilon, data in results.items():
    basis = data['basis']
    c_list = data['c_list']
    f_funcs = [sp.lambdify(t, sum(c_list[j] * basis[j] for j in range(i + 1)), 'numpy')
               for i in range(len(c_list))]
    indices = list(range(0, len(f_funcs), 2)) + [len(f_funcs)-1]
    indices = sorted(set(indices))
    for idx in indices:
        plt.figure(figsize=(6, 4))
        plt.plot(t_vals, y_vals, label='y(t)')
        y_approx = f_funcs[idx](t_vals)
        if np.ndim(y_approx) == 0:
            y_approx = np.full_like(t_vals, y_approx)
        plt.plot(t_vals, y_approx, '--', label=f'S_{idx}(t)')
        plt.title(f'ε={epsilon}, S_{idx}(t)')
        plt.xlabel('t')
        plt.ylabel('y, S_n')
        plt.legend()
        plt.grid(True)

plt.show()
