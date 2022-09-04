import numpy as np
from scipy import special
import matplotlib.pyplot as plt

x = np.linspace(0.1, 20, num=1000)


def k_1n_1(x, C1=1, C2=2):
    return C1 + x * np.log(x) * C2


def k_1n0(x, C1=1, C2=2, A=1, mu0=10**(-1), r0=1, xi=2):
    return C1 + x * np.log(x) * C2 - (A*mu0*r0**(-xi))/(xi+2)**2 * x**(xi+2)


def k0n_1(x, C1=1, C2=2, B=1, mu0=10**(-1)):
    return C1 + x * np.log(x) * C2 - (B*mu0**2)/4 * x**2


def k0n0(x, C1=1, C2=2, A=2, B=2, mu0=10**(-1), r0=1, xi=2):
    return ((C1 + x * np.log(x) * C2 - (B*mu0**2) * x**2) * (xi+2)**2 - 4*A*mu0*r0**(-xi) * x**(xi+2)) / (4*((xi+2)**2))


def k1n_1(x, C1=1, C2=2, B=2, mu0=1):
    return C1 * special.jv(0, x * mu0 * np.sqrt(2*B)) + C2 * special.yn(0, x * mu0 * np.sqrt(2*B))


y_1_1 = k_1n_1(x)
y_10 = k_1n0(x)
y0_1 = k0n_1(x)
y00 = k0n0(x)
y1_1 = k1n_1(x)

labels = ["k=n=-1", "k=-1, n=0", "k=0, n=-1", "k=n=0", "k=1, n=-1"]
functions = [k_1n_1, k_1n0, k0n_1, k0n0, k1n_1]
constant_values = [
    [{"C1": 1, "C2": 1}, {"C1": 1, "C2": 2}, {"C1": 1, "C2": 3}, {"C1": 1, "C2": 4}],
    [{"xi": 0.2}, {"xi": 0.8}, {"xi": 1.2}, {"xi": 1.8}, {"xi": 2.2}],
    [{"B": 1}, {"B": 10}, {"B": 100}, {"B": 1000}],
    [{"xi": 0.2}, {"xi": 0.8}, {"xi": 1.2}, {"xi": 1.8}, {"xi": 2.2}],
    [{"B": 1}, {"B": 2}, {"B": 3}],
]
custom_y_lims = [
    (-10, 200), (-500, 150), (-100, 100), (-1000, 100), (-1, 1)
]
custom_label_variables = [
    "C2", "xi", "B", "xi", "B"
]

fig, axs = plt.subplots(5, figsize=[14, 12], sharex=True)
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)
for k, ax in enumerate(axs):
    for val in constant_values[k]:
        y = functions[k](x, **val)
        ax.plot(x, y, label=f"{custom_label_variables[k]}={val[custom_label_variables[k]]}")
    ax.set_title(labels[k])
    ax.legend(fontsize=14)
    ax.set_ylim(*custom_y_lims[k])
    ax.grid()
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("R(r)", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)

# plt.show()
plt.savefig("azimutal_ode.pdf")
