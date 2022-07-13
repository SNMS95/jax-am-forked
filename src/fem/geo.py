import numpy as onp
import jax
import jax.numpy as np
import os
import gmsh
import meshio
import sys
import time

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)

global_args = {}
global_args['dim'] = 3
global_args['num_quads'] = 8
global_args['num_nodes'] = 8


def gmsh_mesh(mesh_filepath):
    '''
    References:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/hex.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t1.py
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/gmsh_4_7_1/tutorial/python/t3.py
    '''
    offset_x = 0.
    offset_y = 0.
    offset_z = 0.
    domain_x = 1.
    domain_y = 1.
    domain_z = 1.

    global_args['domain_x'] = domain_x

    Nx = 50
    Ny = 50
    Nz = 50

    hx = domain_x / Nx
    hy = domain_y / Ny
    hz = domain_z / Nz

    # Whay divided by two? Because we use [-1, 1] isoparametric elements
    global_args['alpha'] = onp.array([hx/2., hy/2., hz/2.])

    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # save in old MSH format
    Rec2d = True  # tris or quads
    Rec3d = True  # tets, prisms or hexas
    p = gmsh.model.geo.addPoint(offset_x, offset_y, offset_z)
    l = gmsh.model.geo.extrude([(0, p)], domain_x, 0, 0, [Nx], [1])
    s = gmsh.model.geo.extrude([l[1]], 0, domain_y, 0, [Ny], [1], recombine=Rec2d)
    v = gmsh.model.geo.extrude([s[1]], 0, 0, domain_z, [Nz], [1], recombine=Rec3d)

    # print(v)
    # print(s)
  
    gmsh.model.geo.synchronize()

    pg_dim2_bottom = gmsh.model.addPhysicalGroup(2, [s[1][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_bottom, "z0")

    pg_dim2_top = gmsh.model.addPhysicalGroup(2, [v[0][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_top, "z1")

    pg_dim2_front = gmsh.model.addPhysicalGroup(2, [v[2][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_front, "y0")

    pg_dim2_right = gmsh.model.addPhysicalGroup(2, [v[3][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_right, "x1")

    pg_dim2_back = gmsh.model.addPhysicalGroup(2, [v[4][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_back, "y1")

    pg_dim2_left = gmsh.model.addPhysicalGroup(2, [v[5][1]])
    gmsh.model.setPhysicalName(2, pg_dim2_left, "x0")

    pg_dim3_body = gmsh.model.addPhysicalGroup(3, [v[1][1]])
    gmsh.model.setPhysicalName(3, pg_dim3_body, "body")


    gmsh.model.mesh.generate(3)
    gmsh.write(mesh_filepath)
    gmsh.finalize()


def generate_domain():
    mesh_filepath = "post-processing/msh/domain.msh"
    
    generate = True
    if generate:
        gmsh_mesh(mesh_filepath)
  
    mesh = meshio.read(mesh_filepath)
    points = mesh.points # (num_dofs, dim)
    cells =  mesh.cells_dict['hexahedron'] # (num_cells, num_nodes)
    # print(points)
    # print(cells)
    # print(onp.take(points, cells[1], axis=0))
    return mesh


def fem_pre_computations(mesh):
    """Many quantities can be pre-computed and stored for better performance.
    """
    def get_quad_points():
        quad_degree = 2
        quad_points = []
        for i in range(quad_degree):
            for j in range(quad_degree):
                for k in range(quad_degree):
                   quad_points.append([(2*(k % 2) - 1) * np.sqrt(1./3.), 
                                       (2*(j % 2) - 1) * np.sqrt(1./3.), 
                                       (2*(i % 2) - 1) * np.sqrt(1./3.)])
        quad_points = np.array(quad_points) # (quad_degree^dim, dim)
        return quad_points

    def get_shape_val_functions():
        '''
        Hard-coded first order shape functions in the parent domain.
        Important: f1-f8 order must match "cells" in gmsh file 
        '''
        f1 = lambda x: -1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] - 1)
        f2 = lambda x: 1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] - 1)
        f3 = lambda x: -1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] - 1) 
        f4 = lambda x: 1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] - 1)
        f5 = lambda x: 1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] + 1)
        f6 = lambda x: -1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] + 1)
        f7 = lambda x: 1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] + 1)
        f8 = lambda x: -1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] + 1)
        return [f1, f2, f3, f4, f5, f6, f7, f8]

    def get_shape_grad_functions():
        shape_fns = get_shape_val_functions()
        return [jax.grad(f) for f in shape_fns]

    def get_body_force():
        """Pre-compute right-hand-side body force
        In a more general case, this should return ndarray (num_cells, num_quads)
        """
        return 10.

    def get_JxW():
        """Pre-compute Jacobian * weight
        In a more general case, this should return ndarray (num_cells, num_quads)
        """
        return np.prod(global_args['alpha'])

    def get_shape_vals():
        """Pre-compute shape function values

        Returns
        -------
        shape_vals: ndarray
           (8, 8) = (num_quads, num_nodes)  
        """
        shape_val_fns = get_shape_val_functions()
        quad_points = get_quad_points()
        shape_vals = []
        for quad_point in quad_points:
            physical_shape_vals = []
            for shape_val_fn in shape_val_fns:
                physical_shape_val = shape_val_fn(quad_point) 
                physical_shape_vals.append(physical_shape_val)
     
            shape_vals.append(physical_shape_vals)

        shape_vals = np.array(shape_vals) # (num_quads, num_nodes, dim)
        assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
        return shape_vals

    def get_shape_grads():
        """Pre-compute shape function gradients

        Returns
        -------
        shape_grads: ndarray
           (8, 8, 3) = (num_quads, num_nodes, dim)  
        """
        shape_grad_fns = get_shape_grad_functions()
        quad_points = get_quad_points()
        shape_grads = []
        for quad_point in quad_points:
            physical_shape_grads = []
            for shape_grad_fn in shape_grad_fns:
                # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
                # Page 147, Eq. (3.9.3)
                physical_shape_grad = shape_grad_fn(quad_point) / global_args['alpha']
                physical_shape_grads.append(physical_shape_grad)
     
            shape_grads.append(physical_shape_grads)

        shape_grads = np.array(shape_grads) # (num_quads, num_nodes, dim)
        assert shape_grads.shape == (global_args['num_quads'], global_args['num_nodes'], global_args['dim'])
        return shape_grads

    def compute_residual(dofs):
        """Assembly of weak form
        """
        # (num_cells, 1, num_nodes, 1) * (1, num_quads, num_nodes, dim) -> (num_cells, num_quads, num_nodes, dim) 
        u_grads = np.take(dofs, cells)[:, None, :, None] * shape_grads[None, :, :, :] 
        u_grads = np.sum(u_grads, axis=2, keepdims=True) # (num_cells, num_quads, 1, dim) 
        v_grads = shape_grads[None, :, :, :] # (1, num_quads, num_nodes, dim)
        weak_form = np.sum(u_grads * v_grads * JxW, axis=(1, 3)) # (num_cells, num_nodes)
        res = np.zeros_like(dofs)
        res = res.at[cells].add(weak_form)
        return res

    def get_rhs():
        rhs = np.zeros(global_args['num_dofs'])
        v_vals = np.repeat(shape_vals[None, ...], global_args['num_cells'], axis=0) # (num_cells, num_quads, num_nodes)
        rhs_vals = np.sum(v_vals * body_force * JxW, axis=1).reshape(-1) # (num_cells, num_nodes) -> (num_cells*num_nodes)
        rhs = rhs.at[cells.reshape(-1)].add(rhs_vals)   
        return rhs

    cells = mesh.cells_dict['hexahedron'] 
    global_args['num_cells'] = len(cells)
    global_args['num_dofs'] = len(mesh.points)
    body_force = get_body_force()
    JxW = get_JxW()
    shape_vals = get_shape_vals()
    shape_grads = get_shape_grads()
    rhs = get_rhs()

    return compute_residual, rhs


def operator_to_matrix(operator_fn):
    # TODO: nonlinear problems
    J = jax.jacfwd(operator_fn)(np.zeros(args['num_nodes']))
    return J


def apply_bc(res_fn, rhs, left_inds, right_inds):
    def A_fn(dofs):
        """Apply B.C. conditions
        """
        res = res_fn(dofs)
        res = res.at[left_inds].set(dofs[left_inds], unique_indices=True)
        res = res.at[right_inds].set(dofs[right_inds], unique_indices=True)
        return res
    
    b = rhs.at[left_inds].set(0.)
    b = b.at[right_inds].set(0.)

    return A_fn, b


def poisson_solver():
    mesh = generate_domain()

    EPS = 1e-5
    left_inds = onp.argwhere(mesh.points[:, 0] < EPS).reshape(-1)
    right_inds = onp.argwhere(mesh.points[:, 0] >  global_args['domain_x'] - EPS).reshape(-1)

    res_fn, rhs = fem_pre_computations(mesh)
    print("Done pre-computing")
 
    A_fn, b = apply_bc(res_fn, rhs, left_inds, right_inds)

    for i in range(1):
        start = time.time()
        sol, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # gmres, bicgstab, cg
        end = time.time()
        print(f"The {i + 1}th solve took {end - start}")

    print(f"res l_2 = {np.linalg.norm(A_fn(sol) - b)}")  
    print(f"res l_inf = {np.max(np.absolute(A_fn(sol) - b))}")  
    print(f"max of sol = {np.max(sol)}")

    mesh.point_data['sol'] = onp.array(sol, dtype=onp.float32)
    mesh.write(f"post-processing/vtk/fem/sol.vtu")


if __name__ == "__main__":
    # generate_domain()
    poisson_solver()