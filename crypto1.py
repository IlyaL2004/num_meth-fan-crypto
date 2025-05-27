#!/usr/bin/env python3


import time
import argparse
import sympy as sp
from tinyec import ec


def modular_sqrt(a, p):
    if a == 0:
        return 0
    if pow(a, (p-1)//2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p+1)//4, p)
    s, q = 0, p-1
    while q % 2 == 0:
        s += 1
        q //= 2
    z = 2
    while pow(z, (p-1)//2, p) != p-1:
        z += 1
    m, c = s, pow(z, q, p)
    t, r = pow(a, q, p), pow(a, (q+1)//2, p)
    while t != 1:
        i, temp = 1, pow(t, 2, p)
        while temp != 1:
            temp = pow(temp, 2, p)
            i += 1
        b = pow(c, 2**(m-i-1), p)
        m = i
        c = pow(b, 2, p)
        t = t * c % p
        r = r * b % p
    return r


def order_bruteforce(P, max_k, show_progress=False):
    Q = P
    start = time.time() if show_progress else None
    step = max(1, max_k // 100)
    for k in range(1, max_k+1):
        if Q.x is None and Q.y is None:
            if show_progress:
                print()
            return k
        Q += P
        if show_progress and k % step == 0:
            elapsed = time.time() - start
            percent = (k / max_k) * 100
            eta = elapsed * (100 / percent - 1)
            print(f"\rProgress: {percent:5.1f}% | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", end='', flush=True)
    if show_progress:
        print()
    return None


def find_suitable_p(a, b, x, target_seconds=600):
    print("=== Авто-подбор p под ~600 с перебора ===")
    p = 1000003
    while True:
        rhs = (x**3 + a*x + b) % p
        y = modular_sqrt(rhs, p)
        if y is not None:
            field = ec.SubGroup(p, (x, y), None, None)
            curve = ec.Curve(a, b, field)
            P = curve.g
            break
        p = sp.nextprime(p + 1)

    print(f"  Probe@p={p}: 10000 iters...")
    start = time.time()
    order_bruteforce(P, 10000)
    elapsed = time.time() - start
    time_per_iter = elapsed / 10000
    est_iters = int(target_seconds / time_per_iter)
    next_p = sp.nextprime(est_iters)

    print(f"    {10000} iters in {elapsed:.3f}s → {time_per_iter:.6f}s/iter")
    print(f"    Est full iters={est_iters}, suggest p={next_p}")

    p = next_p
    while True:
        rhs = (x**3 + a*x + b) % p
        y = modular_sqrt(rhs, p)
        if y is not None:
            print(f"Найдено подходящее p = {p}")
            # возвращаем p, max_k и y
            return p, int(est_iters * 1.05), y
        p = sp.nextprime(p + 1)


def main():
    parser = argparse.ArgumentParser(description="Полный перебор порядка точки на ЭК")
    parser.add_argument('--p', type=int, help='Поле GF(p)')
    parser.add_argument('--a', type=int, default=2, help='Коэффициент a')
    parser.add_argument('--b', type=int, default=3, help='Коэффициент b')
    parser.add_argument('--x', type=int, default=1, help='x-координата базовой точки')
    parser.add_argument('--max_k', type=int, help='Максимальный порядок для перебора')
    parser.add_argument('--runs', type=int, default=1, help='Количество прогонов')
    parser.add_argument('--progress', action='store_true', help='Показывать прогресс')
    args = parser.parse_args()

    a, b, x = args.a, args.b, args.x

    if args.p is None:
        p, max_k, y = find_suitable_p(a, b, x)
    else:
        p = args.p
        max_k = args.max_k
        if max_k is None:
            raise ValueError("При указании --p нужно задать --max_k")
        rhs = (x**3 + a*x + b) % p
        y = modular_sqrt(rhs, p)
        while y is None:
            print(f"Для x={x} нет корня в GF({p}), подбираем следующее простое...")
            p = sp.nextprime(p + 1)
            rhs = (x**3 + a*x + b) % p
            y = modular_sqrt(rhs, p)
        print(f"Найдено подходящее p = {p}")

    field = ec.SubGroup(p, (x, y), None, None)
    curve = ec.Curve(a, b, field)
    P = curve.g

    print(f"\n=== Кривая: y² = x³ + {a}x + {b} над GF({p})")
    print(f"Базовая точка P = ({P.x}, {P.y})")
    print(f"Будем перебирать до max_k = {max_k}\n")

    times = []
    for run in range(args.runs):
        print(f"--- Полный перебор: прогон {run+1}/{args.runs} ---")
        start = time.time()
        k = order_bruteforce(P, max_k, show_progress=args.progress)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"\nРезультат: порядок = {k}, время = {elapsed:.2f}s\n")

    avg_time = sum(times) / len(times)
    print("=== Финальный результат ===")
    print(f"Порядок точки P = {k}")
    print(f"Среднее время ({args.runs} запусков) = {avg_time:.2f}s")


if __name__ == "__main__":
    main()
