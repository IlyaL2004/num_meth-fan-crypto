import numpy as np
import matplotlib.pyplot as plt

# Параметр a = 4
a = 4

# Диапазон значений x1
x1 = np.linspace(-5, 8, 500)

# Первое уравнение: x2 = 64 / (x1^2 + 16)
x2_1 = 64 / (x1**2 + 16)

# Второе уравнение: окружность (x1 - 2)^2 + (x2 - 2)^2 = 16
theta = np.linspace(0, 2*np.pi, 100)
x1_circle = 2 + 4 * np.cos(theta)
x2_circle = 2 + 4 * np.sin(theta)

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(x1, x2_1, label=r'$(x_1^2 + 16)x_2 = 64$', color='blue')
plt.plot(x1_circle, x2_circle, label=r'$(x_1 - 2)^2 + (x_2 - 2)^2 = 16$', color='red', linestyle='--')

# Настройки графика
plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title('График системы уравнений при $a=4$', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(-5, 8)
plt.ylim(-5, 8)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.show()