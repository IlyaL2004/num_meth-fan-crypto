import numpy as np
import math
import cmath


def norm(vec):
    ans = 0
    for num in vec:
        ans += num * num
    ans = np.sqrt(ans)
    return ans


def get_householder_matrix(A, col_num):
    n = A.shape[0]
    a = A[:, col_num].copy()

    sign = -1 if a[col_num] < 0 else 1 if a[col_num] > 0 else 0

    # Вычисление вектора v
    v = np.zeros(n)
    v[col_num] = a[col_num] + sign * norm(a[col_num:])
    for i in range(col_num + 1, n):
        v[i] = a[i]

    v = v.reshape(n, 1)


    vTv = v.T @ v

    H = np.eye(n) - (2 * v @ v.T) / vTv
    return H


def QR_decomposition(A):
    n = A.shape[0]
    Q = np.eye(n)
    A_i = np.copy(A)

    for i in range(n - 1):
        H = get_householder_matrix(A_i, i)
        Q = Q @ H
        A_i = H @ A_i
    return Q, A_i



def get_roots(A, i):
    n = len(A)

    # Извлекаем элементы подматрицы 2x2
    #a11 = A[i][i]
    #a12 = A[i][i + 1]
    #a21 = A[i + 1][i]
    #a22 = A[i + 1][i + 1]

    # Коэффициенты квадратного уравнения: λ² + bλ + c = 0
    b = -(A[i][i] + A[i + 1][i + 1])
    c = A[i][i] * A[i + 1][i + 1] - A[i][i + 1] * A[i + 1][i]

    # Вычисляем дискриминант
    discriminant = b ** 2 - 4 * c  # a=1, поэтому 4*a*c = 4*1*c

    # Находим корни
    if discriminant >= 0:
        sqrt_d = math.sqrt(discriminant)
    else:
        sqrt_d = cmath.sqrt(discriminant)

    root1 = (-b + sqrt_d) / 2
    root2 = (-b - sqrt_d) / 2

    return [root1, root2]

def are_roots(lambda1, lambda2, eps):
    # Нет сортировки!
    return (
        abs(lambda1[0] - lambda2[0]) <= eps and
        abs(lambda1[1] - lambda2[1]) <= eps
    )
# Checking if our values are complexZ
def complex(A, i, eps):
    Q, R = QR_decomposition(A)
    A_next = R @ Q
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return are_roots(lambda1, lambda2, eps)

def get_eigen_values_QR(A, eps):
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []
    i = 0

    while i < n:
        # Если это последний элемент, добавляем его и завершаем
        if i == n - 1:
            eigen_values.append(A_i[i, i])
            i += 1
            continue

        # Итерации QR-разложения для текущего i
        while True:
            Q, R = QR_decomposition(A_i)
            A_i = R @ Q

            # Проверка на выход за границы при обращении к A_i[i+1:, i]
            if i + 1 >= n:
                break  # Предотвращаем ошибку индексации

            # Проверка вещественного собственного значения
            sub_column = A_i[i + 1:, i]
            if norm(sub_column) <= eps:
                eigen_values.append(A_i[i, i])
                i += 1
                break

            # Проверка на комплексные значения (только если i+2 < n)
            if i + 2 < n:
                sub_sub_column = A_i[i + 2:, i]
                if norm(sub_sub_column) <= eps and complex(A_i, i, eps):
                    roots = get_roots(A_i, i)
                    eigen_values.extend(roots)
                    i += 2
                    break
            else:
                # Если i+2 >= n, обрабатываем как вещественное значение
                eigen_values.append(A_i[i, i])
                i += 1
                break

    return eigen_values


if __name__ == '__main__':

    A = [[5, -5, -6],
          [-1, -8, -5],
          [2, 7, -3]]

    A = np.array(A, dtype='float')
    eps = 0.000001

    eig_values = get_eigen_values_QR(A, eps)
    print('Eigen values:', eig_values)