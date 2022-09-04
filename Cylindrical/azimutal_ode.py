
from sympy import E, Eq, Function, Derivative as D, exp, diff, classify_pde, checkpdesol, ln, cosh, sqrt, oo
from sympy import dsolve
from sympy.abc import x, z, chi, zeta, r
from random import random, choice
from sympy import *
import numpy as np

u = symbols(r"\mu_0")
A = symbols("A")
B = symbols("B")
r0 = symbols("r_0")
xi = symbols(r"\xi")

y = Function("R")(r)

for k in [-3, -2, -1, 0, 1, 2, 3]:
    for n in [-1, 0, 1, 2]:
        print(f"Resolvendo problema para $k={k}$ e $n={n}$.")
        ode = Eq(r * D(y, r, 2) + D(y, r, 1) + r * (k+1) * u**2 * B * y**k, - (n+1) * u * A * r0**(-xi) * r**(1+xi) * y**n)
        print(f"O problema é:\n "
              r"\be"
              f"\n{latex(ode)}\n"
              r"\ee")
        try:
            sol = dsolve(ode)

            func = sol.args[1]
            print(f"A solução é:"
                  rf"\be"
                  f"\n R(r) = {latex(func)}\n"
                  rf"\ee")
        except NotImplementedError:
            print("Infelizmente esse caso não tem solução.\n")



