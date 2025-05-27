import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
f = np.sqrt(1 - x**2)
g = np.exp(x) - 0.1

plt.plot(x, f, label=r'$f(x) = \sqrt{1 - x^2}$')
plt.plot(x, g, label=r'$g(x) = e^x - 0.1$')

plt.xticks(np.linspace(0, 1, 11))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()