import math
import random

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 5 * x * x - 20 * x + 3


def bisection(f, a, b, eps):
    delta = eps / 10
    iterations = 0
    while b - a >= eps:
        mean = (a + b) / 2
        x1 = mean - delta
        x2 = mean + delta
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        iterations += 1
    # print("iterations: " + str(iterations))
    # print("function calls: " + str(iterations * 2))
    return (a + b) / 2, iterations, iterations * 2


def golden_section(f, a, b, eps):
    phi = (3 - math.sqrt(5)) / 2
    t = phi * (b - a)
    x1 = a + t
    x2 = b - t
    fx1 = f(x1)
    fx2 = f(x2)
    iterations = 0
    while b - a >= eps:
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + phi * (b - a)
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = b - phi * (b - a)
            fx2 = f(x2)
        iterations += 1
    # print("iterations: " + str(iterations))
    # print("function calls: " + str(iterations + 2))
    return (a + b) / 2, iterations, iterations + 2


def fibonacci(f, a, b, eps):
    fib = [1, 1]
    n = 1
    while fib[n] <= (b - a) / eps:
        fib.append(fib[n] + fib[n - 1])
        n += 1
    n -= 2
    x1 = a + fib[n] / fib[n + 2] * (b - a)
    x2 = a + fib[n + 1] / fib[n + 2] * (b - a)
    fx1 = f(x1)
    fx2 = f(x2)
    k = 0
    while k < n:
        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + (b - x2)
            fx1 = f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + (b - x1)
            fx2 = f(x2)
        k += 1
    # print("iterations: " + str(n))
    # print("function calls: " + str(n + 2))
    return (a + b) / 2, n, n + 2


def grad_desc(x0, f, df, def_alpha):
    x = x0.copy()
    path = []
    iter = 0
    for i in range(100000):
        path.append(x.copy())
        grad = df(x)
        alpha = def_alpha
        if def_alpha == -1:
            cur_f = f(x)
            delta = 1e-8
            next_f = cur_f
            while next_f <= cur_f:
                cur_f = next_f
                next_f = f(x - grad * delta)
                if next_f <= cur_f:
                    delta *= 2
            g = lambda alpha: f(x - grad * alpha)
            alpha, _, _ = bisection(g, 0, delta, 1e-5)
        new_x = x - alpha * grad
        iter += 1
        if np.linalg.norm(x - new_x) < 1e-5:
            break
        x = new_x
    return np.array(path), iter


def quad(params):
    cx2, cy2, cxy = params
    return lambda x: cx2 * x[0] * x[0] + cy2 * x[1] * x[1] + cxy * x[0] * x[1]


def dquad(params):
    cx2, cy2, cxy = params
    return lambda x: np.array([2 * cx2 * x[0] + cxy * x[1],
                               2 * cy2 * x[1] + cxy * x[0]])


def quadratic_form_coefs(n, k):
    c = np.zeros(n)
    c[0] = 1
    c[1] = k
    for i in range(2, n):
        c[i] = random.randint(1, k)
    return c


def qf(c):
    return lambda x: np.dot(c, x * x)


def dqf(c):
    return lambda x: 2 * c * x


def plot(f, path, ax, color, label):
    if ax is None:
        _, ax = plt.subplots()
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        Z = np.array([list(map(f, x)) for x in np.dstack((X, Y))])
        ax.set_aspect(1)
        ax.contour(X, Y, Z, 20, colors='b')
        color = 'r'
    path = np.clip(path, -3, 3)
    ax.plot(path[:, 0], path[:, 1], color=color, label=label)
    return ax


def searchers():
    a = -10
    b = 100
    for i in range(8):
        eps = 1 / 10 ** i
        print("eps ==", eps)
        res, iter, f_calls = bisection(f, a, b, eps)
        print("\tbisection: %d iters, %d calls" % (iter, f_calls))
        # print(math.log((b - a) / eps, 2))
        res, iter, f_calls = golden_section(f, a, b, eps)
        print("\tgolden_section: %d iters, %d calls" % (iter, f_calls))
        # print(math.log((b - a) / eps, 2 / (math.sqrt(5) - 1)))
        res, iter, f_calls = fibonacci(f, a, b, eps)
        print("\tfibonacci: %d iters, %d calls" % (iter, f_calls))


def fixed_vs_bisection():
    x = np.array([2.5, 0])
    params = (1, 10, 2)
    q = quad(params)
    dq = dquad(params)
    _, iter_fixed_008 = grad_desc(x, q, dq, 0.08)
    _, iter_fixed_001 = grad_desc(x, q, dq, 0.01)
    _, iter_bisection = grad_desc(x, q, dq, -1)
    print("iter_fixed(0.08):", iter_fixed_008)
    print("iter_fixed(0.01):", iter_fixed_001)
    print("iter_bisection:", iter_bisection)


def some_plots():
    x = np.array([2.5, 1])
    params = (1, 10, 2)
    q = quad(params)
    path, iter = grad_desc(x, q, dquad(params), 0.09)
    ax = plot(q, path, None, 'r', "fixed: 0.09")
    path, iter = grad_desc(x, q, dquad(params), 0.01)
    plot(q, path, ax, 'g', "fixed: 0.01")
    path, iter = grad_desc(x, q, dquad(params), -1)
    plot(q, path, ax, 'y', "bisection")
    plt.legend()
    plt.show()

    x = np.array([2.5, 2])
    params = (1, 9, 6)
    q = quad(params)
    path, iter = grad_desc(x, q, dquad(params), -1)
    plot(q, path, None, 'r', 'bisection')
    plt.legend()
    plt.show()


def nk():
    random.seed(12345)
    for n in range(2, 11):
        for k in range(0, 11):
            c = quadratic_form_coefs(n, 2 ** k)
            path, iter = grad_desc(np.ones(n), qf(c), dqf(c), -1)
            print(iter, end="\t")
        print()


def main():
    searchers()
    fixed_vs_bisection()
    some_plots()
    nk()


if __name__ == '__main__':
    main()
