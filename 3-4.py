def locate_index(arr, target):
    for idx, val in enumerate(arr):
        if val > target:
            return idx
    raise ValueError(f"Значение {target} находится вне диапазона данных.")


def compute_derivatives(nodes, values, x):
    idx = locate_index(nodes, x)

    delta_left = (values[idx - 1] - values[idx - 2]) / (nodes[idx - 1] - nodes[idx - 2])
    delta_right = (values[idx] - values[idx - 1]) / (nodes[idx] - nodes[idx - 1])

    interval_width = nodes[idx] - nodes[idx - 2]

    first_deriv = delta_left + (delta_right - delta_left) / interval_width * (2 * x - nodes[idx - 2] - nodes[idx - 1])
    second_deriv = 2 * (delta_right - delta_left) / interval_width

    return first_deriv, second_deriv


if __name__ == "__main__":
    query_point = 2.0
    node_coords = [1.0, 1.5, 2.0, 2.5, 3.0]
    node_values = [0.0, 0.40547, 0.69315, 0.91629, 1.0986]

    deriv1, deriv2 = compute_derivatives(node_coords, node_values, query_point)

    print(f"Первая производная в точке {query_point}: {deriv1:.6f}")
    print(f"Вторая производная в точке {query_point}: {deriv2:.6f}")