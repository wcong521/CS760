import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error

a = 0
b = 10

n = 100
# n = 15

x_train = np.random.uniform(a, b, n)
y_train = np.sin(x_train)

x_test = np.random.uniform(a, b, n)
y_test = np.sin(x_test)

print("\nLagrange Interpolation\n")

f = lagrange(x_train, y_train)

train_error = np.sqrt(mean_squared_error(y_train, f(x_train)))
print(f"\tTrain Error: {train_error}")

test_error = np.sqrt(mean_squared_error(y_test, f(x_test)))
print(f"\tTest Error: {test_error}")

print("\nLagrange Interpolation (w/ Noise)\n")

for std in np.arange(0, 1, 0.1):
    gaussian_noise = np.random.normal(0, std, n)
    x_train_noisy = x_train + gaussian_noise
    y_train_noisy = np.sin(x_train_noisy)

    f = lagrange(x_train_noisy, y_train_noisy)

    print(f"\tSD: {std}")
    train_error = np.sqrt(mean_squared_error(y_train_noisy, f(x_train_noisy)))
    print(f"\tTrain Error: {train_error}")
    test_error = np.sqrt(mean_squared_error(y_test, f(x_test)))
    print(f"\tTest Error: {test_error}\n")

# x = np.arange(a, b, 0.1)

# plt.scatter(x_train, y_train)
# plt.plot(x, f(x))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Exercise 4')
# plt.show()