import jax.numpy as np
import jax
import os
import glob

import jaxopt

from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.solver import ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.common import rectangle_mesh
from jax.scipy.optimize import minimize

from fem_model import Elasticity

problem_name = 'plate'
root_path = os.path.join(os.path.dirname(__file__), 'data')

files = glob.glob(os.path.join(root_path, f'vtk/{problem_name}/*'))
for f in files:
    os.remove(f)

L = 60.
W = 30.
N_L = 60
N_W = 30
meshio_mesh = rectangle_mesh(N_L, N_W, L, W)
jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['quad'])


def fixed_location(point):
    return np.isclose(point[0], 0., atol=1e-5)


def load_location(point):
    return np.logical_and(np.isclose(point[0], L, atol=1e-5),
                          np.isclose(point[1], 0., atol=1. + 1e-5))


def dirichlet_val(point):
    return 0.


def neumann_val(point):
    return np.array([0., -1])


dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]
neumann_bc_info = [[load_location], [neumann_val]]
problem = Elasticity(jax_mesh,
                     vec=2,
                     dim=2,
                     ele_type='QUAD4',
                     dirichlet_bc_info=dirichlet_bc_info,
                     neumann_bc_info=neumann_bc_info,
                     additional_info=(problem_name, ))

fwd_pred = ad_wrapper(problem, linear=True, use_petsc=False)


def J_fn(dofs, params):
    """J(u, p)
        """
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    compliance = problem.compute_compliance(neumann_val, sol)
    return compliance


def J_total(params):
    """J(u(p), p)
    """
    sol = fwd_pred(params)
    dofs = sol.reshape(-1)
    obj_val = J_fn(dofs, params)
    return obj_val


outputs = []


def output_sol(params, obj_val):
    print("\nOutput solution - need to solve the forward problem again...")
    sol = fwd_pred(params)
    vtu_path = os.path.join(
        root_path, f'vtk/{problem_name}/sol_{output_sol.counter:03d}.vtu')
    save_sol(problem,
             sol,
             vtu_path,
             cell_infos=[('theta', problem.full_params[:, 0])])
    print(f"compliance = {obj_val}")
    outputs.append(obj_val)
    output_sol.counter += 1


output_sol.counter = 0

vf = 0.5


def objectiveHandle(rho):
    J, dJ = jax.value_and_grad(J_total)(rho)
    output_sol(rho, J)
    return J, dJ


# Initialize params
params = vf * np.ones((len(problem.flex_inds), 1))
print(f'Initial compliance is: {J_total(params):.5e}')

# # # First let's see if the function can be jitted
# solver = jaxopt.LBFGS(fun=objectiveHandle,
#                       value_and_grad=True,
#                       has_aux=False,
#                       jit=False,)

params = vf * np.ones((len(problem.flex_inds), 1))
res = minimize(fun=objectiveHandle, x0=params, method='BFGS', options={'maxiter':10})

print(res.fun)
#Trainiting loop
# opt_state = solver.init_state(params)

# for _ in range(2):
#     params, opt_state = solver.update(params=params, state=opt_state)
