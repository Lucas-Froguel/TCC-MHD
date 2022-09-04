from mpmath import *
import matplotlib.pyplot as plt
import numpy as np


x_val = np.linspace(0.01, 1000, num=1000)


def function_sin(x_val):
    y_real = np.zeros(len(x_val))
    y_im = np.zeros(len(x_val))
    for i in range(len(x_val)):
        val = - x_val[i] ** (5/2) * np.sin((np.sqrt(7)/2)*np.log(x_val[i])) * hyp1f2(1, 2-2*j*np.sqrt(2), 2+2*j*np.sqrt(2), -x_val[i]**2)
        y_real[i] = float(val.real) * 1.45*10**6
        y_im[i] = float(val.imag)

    return y_real, y_im


def function_cos(x_val):
    y_real = np.zeros(len(x_val))
    y_im = np.zeros(len(x_val))
    for i in range(len(x_val)):
        val = x_val[i] ** (5/2) * np.cos((np.sqrt(7)/2)*np.log(x_val[i])) * hyp1f2(1, 2-2*j*np.sqrt(2), 2+2*j*np.sqrt(2), -x_val[i]**2)
        y_real[i] = float(val.real) * 1.45*10**6
        y_im[i] = float(val.imag)

    return y_real, y_im


# y_sin, y_i = function_sin(x_val)
# y_cos, y_ii = function_cos(x_val)
#
# plt.plot(x_val, y_sin, label="u")
# plt.plot(x_val, y_cos, label="v")
# # plt.plot(x_val, y_i, label="im")
# plt.legend()
# plt.xlabel("r")
# plt.ylabel("R(r)")
# plt.axis()
# plt.grid()
# plt.show()

