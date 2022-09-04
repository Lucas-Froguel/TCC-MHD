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

init_printing()


class Separable:
    def __init__(self):
        # Define the symbols
        self.u = symbols(r"\mu_0")
        self.R = symbols("R")
        self.T = symbols("T")
        self.grav = symbols("g")
        self.pi = symbols(r"\pi")

        self.I0 = symbols("I_0")
        self.l = symbols("l")
        self.C = symbols("C")
        self.k = symbols("k")

        # These two can be any functions and should be as many as possible
        self.f = Function("X")(x)
        self.g = Function("Z")(z)
        self.exp_h = exp(-(self.grav * z) / (self.R * self.T))

        # Define the function we want to find in raw form
        self.psi = Function("psi")(x, z)
        self.psi_x = Function("psi")(x, z)
        self.psi_z = Function("psi")(x, z)
        self.X = Function("X")(x)
        self.Z = Function("Z")(z)

        self.I = Function("psi")(x, z)
        self.p = Function("psi")(x, z)

        # Define ODEs
        self.x_ode = None
        self.x_sol = None
        self.z_ode = None
        self.z_sol = None

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
        self.scope = [x, z]
        self.num_exp = 2
        self.num_ops = 5

    def define_odes(self):
        if self.ind > 1:
            self.X = Function("X")(x)
            self.Z = Function("Z")(z)
        self.x_ode = Eq(D(self.X, x, 2) + self.u**2 * self.f + self.C * self.X, 0)
        self.z_ode = Eq(D(self.Z, z, 2) + self.u * self.g * self.exp_h - self.C * self.Z, 0)

    def solve_odes(self):
        print("Solving X...")
        self.x_sol = dsolve(self.x_ode)
        self.X = self.x_sol.args[1]
        print("Solving Z...")
        self.z_sol = dsolve(self.z_ode)
        self.Z = self.z_sol.args[1]

    def calculate_psi(self):
        self.psi = self.X * self.Z
        print(f"Psi is:\n psi = {latex(self.psi)}")

    def calculate_partial_derivatives_psi(self):
        print("Calculating partial derivatives...")
        self.psi_x = diff(self.psi, x)
        self.psi_z = diff(self.psi, z)
        print(f"Partial derivatives are:\n psi_x={latex(self.psi_x)}\n psi_z={latex(self.psi_z)}")

    def print_status(self):
        message = rf"""

----------- LOG -----------

It is setup with functions:
    f = {latex(self.f)}
and
    g = {latex(self.g)}\
This means we have:
    \psi = {latex(self.psi)}
which yields the partial derivative:
psi_x={latex(self.psi_x)}
psi_z={latex(self.psi_z)}
"""
        print(message)

    def calculate_and_print(self, print=False, save=True):
        self.define_odes()
        self.solve_odes()
        self.calculate_psi()
        self.calculate_partial_derivatives_psi()
        if print:
            self.print_status()
        if save:
            problem.generate_tex_status()
            problem.write_to_tex()
            problem.end_tex_status()

    def generate_tex_status(self):
        with open("../../template.txt", "r") as txt:
            with open("../../tex_files/separable.tex", "w+") as tex:
                tex.write(txt.read())

    def write_to_tex(self):
        sec = r"{Caso " + str(self.ind) + r"}"
        label_1 = r"{eq:iso_cart_" + str(self.ind) + r":1}"
        label_2 = r"{eq:iso_cart_" + str(self.ind) + r":2}"
        message = rf"""

        \section{sec}
        Esse caso considerará as funções
        \bea
            f &=& {latex(self.f)} \\
            g &=& {latex(self.g)}
        \eea
        Isso significa que precisamos resolver as EDOs:
        \be\label{label_1}
            {latex(self.x_ode)}
        \ee
        E
        \be\label{label_2}
            {latex(self.z_ode)}
        \ee
        Isso resulta na solução parcial:
        \bea
            X &=& {latex(self.X)} \\
            Z &=& {latex(self.Z)}
        \eea
        que implica no perfil:
        \be
            \Psi = {latex(self.psi)}
        \ee
        Isso leva aos campos magnéticos:
        \bea
            B_z &=& {latex(self.psi_x)}\\
            B_x &=& {latex(-self.psi_z)}
        \eea
        """

        with open("../../tex_files/separable.tex", "a") as file:
            file.write(message)

    def end_tex_status(self):
        with open("../../tex_files/separable.tex", "a") as file:
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
        self.define_odes()
        self.solve_odes()
        self.calculate_psi()
        self.calculate_partial_derivatives_psi()

    def run(self):
        # expressions = self.generate_random_expressions()
        expressions = [
            (0, 0),
            (1, 1),
            (x, 0),
            (self.X, 0),
            (1 + x + x**2 + x**3, 0),
            (x**4, 0),
            (self.X, self.Z),
            (sin(x) + cos(x), 0),
        ]
        self.generate_tex_status()
        # self.calculate()
        # self.write_to_tex()
        for exp in expressions:
            print(f"Iteration {self.ind}...\n{exp}")
            self.f = exp[0]
            self.g = exp[1]
            try:
                self.calculate()
            except Exception as e:
                print(f"\n ---------Error---------\nFailed iteration for expressions:\n{exp}")
                pass
            self.write_to_tex()
            self.ind += 1

        self.end_tex_status()


problem = Separable()
# problem.calculate_and_print()
# problem.f = 1 + x**2 + problem.X
# problem.g = 0
# problem.calculate_and_print()
problem.run()
