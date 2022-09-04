from sympy import E, Eq, Function, Derivative as D, exp, diff, classify_pde, checkpdesol, ln, cosh, sqrt, oo
from sympy import dsolve
from sympy.abc import x, z, xi, zeta
from sympy import symbols
from sympy import init_printing, latex
from sympy.integrals import integrate, Integral
from sympy import sin, cos, sinh, cosh, tan, tanh, cot, coth, sec, sech, csc, csch, asin, asinh,\
    acos, acosh, atan, atanh, acot, acoth, asec, asech, acsc, acsch
from sympy.parsing.sympy_parser import parse_expr, T
from random import random, choice
import time
from function_timeout import exit_after

init_printing()


class GreenPDE:
    def __init__(self):
        # Define the symbols
        self.u = symbols(r"\mu_0")
        self.R = symbols("R")
        self.T = symbols("T")
        self.g = symbols("g")
        self.pi = symbols(r"\pi")

        self.I0 = symbols("I_0")
        self.l = symbols("l")
        self.C = symbols("C")
        self.k = symbols("k")

        # These two can be any functions and should be as many as possible
        # xi e zeta are the x' and z' for the green function
        self.f = exp(-(xi**2 + zeta**2))
        self.h = 0
        self.exp_h = exp(-(self.g * zeta) / (self.R * self.T))

        # Define green function for psi
        self.G = (1/(2*self.pi)) * ln(sqrt((x-xi)**2 + (z-zeta)**2))

        # Define the function we want to find in raw form
        self.psi = Function("psi")(x, z)
        self.psi_x = Function("psi")(x, z)
        self.psi_z = Function("psi")(x, z)

        self.I = Function("psi")(x, z)
        self.p = Function("psi")(x, z)
        self.int = Function("int")(x, xi, z, zeta)

        # Define global index
        self.ind = 1

        # Path
        self.tex_path = "../../tex_files/green.tex"

        # Random expressions generation
        # self.UNARIES = [sqrt, exp, ln, sin, cos, sinh, cosh, tan, tanh, cot, coth, sec, sech, csc, csch,
        #                 asin, asinh, acos, acosh, atan, atanh, acot, acoth, asec, asech, acsc, acsch]
        self.UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
                   "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
                   "atan(%s)", "-%s"]
        # self.BINARIES = [+, -, *, /, **]
        self.BINARIES = ["%s + %s", "%s - %s", "%s * %s", "%s / %s", "%s ** %s"]

        self.PROP_PARENTHESIS = 0.3
        self.PROP_BINARY = 0.7
        self.scope = [xi, zeta]
        self.num_exp = 2
        self.num_ops = 5

    def integrate_psi(self):
        print("Integrating psi...")
        self.int = self.G * (self.u**2 * self.f + self.u * self.h * self.exp_h)
        print(f"Function to integrate is:\n {latex(self.int)}")
        self.psi = - integrate(self.int, (xi, -oo, oo), (zeta, 0, oo))

        print(f"Result of integration of psi is:\n psi={latex(self.psi)}")

    def calculate_partial_derivatives_psi(self):
        print("Calculating partial derivatives...")
        self.psi_x = diff(self.psi, x)
        self.psi_z = diff(self.psi, z)
        print(f"Partial derivatives are:\n psi_x={latex(self.psi_x)}\n psi_z={latex(self.psi_z)}")

    def integrate_I(self):
        print("Integrating I...")
        # print(f"Function to integrate is:\n {latex()}")
        self.I = 2 * (integrate(self.f * self.psi_x, x) + integrate(self.f * self.psi_z, z))
        print(f"Result of integration of I is:\n {latex(self.I)}")

    def integrate_p(self):
        print("Integratin p...")
        self.p = 2 * (integrate(self.h * self.psi_x, x) + integrate(self.h * self.psi_z, z))
        print(f"Result of integration of p is:\n {latex(self.p)}")

    def print_status(self):
        message = rf"""

----------- LOG -----------

Our problem has the green function
    G = {latex(self.G)}
It is setup with functions:
    f = {latex(self.f)}
and
    g = {latex(self.h)}\
This means we have to integrate:
    \psi = {latex(- Integral(self.int, (xi, -oo, oo), (zeta, 0, oo)))}
which yields the partial derivative
The results are:
\psi = {latex(self.psi)}
and
I = {latex(self.I)}
and
p = {latex(self.p)}
"""
        print(message)

    def calculate_and_print(self, print=True, save=True):
        self.integrate_psi()
        self.calculate_partial_derivatives_psi()
        self.integrate_I()
        self.integrate_p()
        if print:
            self.print_status()
        if save:
            problem.generate_tex_status()
            problem.write_to_tex()
            problem.end_tex_status()

    def generate_tex_status(self):
        with open("../../template.txt", "r") as txt:
            with open("../../tex_files/green.tex", "w+") as tex:
                tex.write(txt.read())

    def write_to_tex(self):
        sec = r"{Caso " + str(self.ind) + r"}"
        message = rf"""

        \section{sec}
        Nosso problema tem a função de Green
        \be
            G = {latex(self.G)}\
        \ee
        Consideraremos as funções:
        \bea
            f &=& {latex(self.f)} \\
            g &=& {latex(self.h)}
        \eea
        Então, precisamos integrar
        \be
            \psi = {latex(- Integral(self.int, (xi, -oo, oo), (zeta, 0, oo)))}
        \ee
        Isso resulta em
        \be
            \psi = {latex(self.psi)}
        \ee
        com derivadas parciais:
        \bea
            \partial_x\psi &=& {latex(self.psi_x)} \\
            \partial_z\psi &=& {latex(self.psi_z)}
        \eea
        Por fim, a corrente a pressão são:
        \bea
            I &=& {latex(self.I)} \\
            p &=& {latex(self.p)}
        \eea
        """

        with open("../../tex_files/green.tex", "a") as file:
            file.write(message)

    def end_tex_status(self):
        with open("../../tex_files/green.tex", "a") as file:
            file.write(r"\end{document}")

    def generate_random_expressions(self):
        expressions = []
        for _ in range(self.num_exp):
            scope = list(self.scope)  # make a copy first, append as we go
            for _ in range(self.num_ops):
                if random() < self.PROP_BINARY:  # decide unary or binary operator
                    ex = choice(self.BINARIES) % (choice(scope), choice(scope))
                    if random() < self.PROP_PARENTHESIS:
                        ex = "(%s)" % ex
                    scope.append(ex)
                else:
                    scope.append(choice(self.UNARIES) % choice(scope))
                expressions.append(parse_expr(scope[-1], local_dict={"xi": xi, "zeta": zeta}, transformations="all"))
        return expressions

    # @exit_after(60)
    def calculate(self):
        self.integrate_psi()
        self.calculate_partial_derivatives_psi()
        self.integrate_I()
        self.integrate_p()

    def run(self):
        expressions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (xi, zeta),
            (xi ** 4, 1),
            (1 + xi + xi ** 2 + xi ** 3, 0),
            (sin(xi), cos(zeta)),
        ]
        self.generate_tex_status()
        for exp in expressions:
            print(f"Iteration {self.ind}...")
            self.f = exp[0]
            self.h = exp[1]
            try:
                self.calculate()
            except Exception as e:
                print(f"\n ---------Error---------\nFailed iteration for expression:\n{exp}")
                pass
            self.write_to_tex()
            self.ind += 1

        self.end_tex_status()


problem = GreenPDE()
# problem.calculate_and_print()
problem.run()
