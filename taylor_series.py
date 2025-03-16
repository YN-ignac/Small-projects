import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

def taylor_series(func, var, a, n):
    """
    Function for Taylor expansion

    func :  sympy expresion
            Function we want to evaluate
    var :   sympy symbol
            Variable by which we derive
    a :     number (real)
            The point around which Taylor series develops
    n :     number (integer)
            Number of terms in the polynom
    """
    
    taylor_expansion = sum(
        (func.diff(var,i).subs(var,a) / sp.factorial(i)) * (var-a)**i
        for i in range(n+1)
    )
    return taylor_expansion

x = sp.Symbol('x')
fun = x**2 + sp.sin(x)
a = 0 
n = 10

taylor_approx = taylor_series(fun, x, a, n)

print("Taylor's series: ", taylor_approx)
    
