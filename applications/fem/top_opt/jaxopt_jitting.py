import jax
#import numpy as np
from jax import custom_vjp
from jaxopt import LBFGS, GradientDescent
import jax.numpy as np


@custom_vjp
def f(x):
    # The function deliberately uses numpy functions which are not jittable
    return -(x[0] + np.sin(x[0])) * np.exp(-x[0]**2.0)


def f_fwd(x):
    return f(x), (x, )


def f_bwd(x, g):
    return -(1.0 + np.cos(x[0])) * np.exp(-x[0]**2.0) - 2.0 * x[0] * (
        x[0] + np.sin(x[0])) * np.exp(-x[0]**2.0),


f.defvjp(f_fwd, f_bwd)

print_errors = False  # Flag for printing the errors

# Check if the custom function is jittable
try:
    f_jitted = jax.jit(f)
    f_jitted(np.array([1.0]))
    print("Function is jittable")
except Exception as e:
    print("Function is not jittable")
    if print_errors:
        print("Error: ", e)

# Check if the custom function is differentiable
try:
    f_grad = jax.grad(f)
    print("Function is differentiable")
except Exception as e:
    print("Function is not differentiable")
    if print_errors:
        print("Error: ", e)

# Run the solver (optimal input is around 0.679579)
solver = GradientDescent(fun=f, jit=True, maxiter=1000)
res = solver.run(np.array([0.6]))


print(res.params)
