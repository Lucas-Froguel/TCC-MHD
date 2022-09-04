import numpy as np
import matplotlib.pyplot as plt


x_val = np.linspace(0.1, 1000, num=1001)


mu0 = 1
B = 0
max_p = 10


def calculate_coeff(p, mu0=1, B=1):
    val = 1
    for k in range(1, p+1):
        val = val * (-1)**p / (k**2 + 8) * np.sqrt(mu0**4*B**2 + 7*(mu0**2*B - k*np.sqrt(7))**2 / (4*k**2))

    return val


def calculate_trig_coeff(p, mu0=1, B=1):
    val = 0
    for j in range(1, p+1):
        if j == 2:
            print(1)
        param = (np.sqrt(7) / (2*j)) * ((j*np.sqrt(7)) / (mu0**2 * B) - 1)
        val += np.arctan(param)

    return val


def calculate_exp(x_val, max_p, mu0=1, B=1):
    cos_val = []
    sin_val = []
    for p in range(1, max_p+1):
        coeff = calculate_coeff(p, mu0=mu0, B=B)
        trig = calculate_trig_coeff(p, mu0=mu0, B=B)
        sin_val.append(coeff * np.sin(trig))
        cos_val.append(coeff * np.cos(trig))

    x_sin = np.zeros(len(x_val))
    x_cos = np.zeros(len(x_val))
    for i, x in enumerate(x_val):
        sin = 0
        cos = 0
        for p in range(max_p):
            sin += sin_val[p] * x**(p+1)
            cos += cos_val[p] * x**(p+1)
        x_sin[i] = sin
        x_cos[i] = cos

    return x_sin, x_cos


def calculate_u_v(x, max_p, mu0=1, B=1):
    x_sin, x_cos = calculate_exp(x, max_p, mu0=mu0, B=B)

    u = x * np.cos(np.sqrt(7) * np.log(x)) * x_cos + x * np.sin(np.sqrt(7) * np.log(x)) * x_sin
    v = x * np.sin(np.sqrt(7) * np.log(x)) * x_cos - x * np.cos(np.sqrt(7) * np.log(x)) * x_sin

    return u, v


u, v = calculate_u_v(x_val, max_p, mu0=mu0, B=B)

plt.plot(x_val, u, label="u")
plt.plot(x_val, v, label="v")
plt.legend()
plt.show()
