import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression, ElasticNet

from hypergeometric import function_cos, function_sin


def ode(t, y, mu0, B, A, l, a):
    f, g = y
    dydt = [
        g, -(mu0 ** 2 * B + 2 / t**2) * f  # + mu0 * A * t**2 * np.exp(l * (t / a) ** 2)
    ]
    return dydt


mu0 = 10**(-4)
B = 0
A = 0.
a = 200

t_eval = np.linspace(0.01, 1000, num=1000)

y0 = [0., 0.5]
for l in [0.9]:
    print("Solving numerically...")
    sol = solve_ivp(ode, (t_eval.min(), t_eval.max()), y0, args=(mu0, B, A, l, a), t_eval=t_eval)
    plt.plot(t_eval, sol.y[0, :], label=f'numeric')

y = np.array(sol.y[0, :])

print("Calculating u and v...")
y_sin, y_i = function_sin(t_eval)
y_cos, y_ii = function_cos(t_eval)
arr = np.array([[y1, y2] for y1, y2 in zip(y_sin, y_cos)])

print("Doing linear regression...")
reg = LinearRegression(fit_intercept=False).fit(arr, y)
print(reg.coef_)


plt.plot(t_eval, 10**(-9) * y_sin, label="u")
plt.plot(t_eval, 10**(-9) * y_cos, label="v")
# plt.plot(t_eval, y_sin + y_cos, label="u+v")
plt.plot(t_eval, reg.predict(arr), label="lin-reg")
plt.legend()
plt.xlabel("r", fontsize=16)
plt.ylabel("R(r)", fontsize=16)
plt.axis()
plt.grid()
# plt.ylim(-10, 10)
plt.legend(fontsize=16)
# plt.savefig("uv-fit.pdf")
#
plt.show()