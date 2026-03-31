"""
Problem setup: mesh, spaces, and operator assembly for the parametric DPG system.
"""

import ngsolve as ng
from ngsolve import dx, grad, x, y, exp
from ngsolve.meshes import MakeStructured2DMesh

from pymor.bindings.ngsolve import NGSolveMatrixOperator, NGSolveVectorSpace
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

from utils_stationary import RectChar
from fom_stationary import DPGModel

# =============================================================================
# 1. MESH GENERATION AND FUNCTION SPACE CONSTRUCTION
# =============================================================================
def _create_mesh_and_spaces(maxh, p):
    """
    Create NGSolve function spaces for DPG formulation.
    """
    N = int(1 / maxh)
    if N % 2 != 0: N += 1

    print(f"  > Generating Structured Mesh ({N}x{N})...")
    mesh= MakeStructured2DMesh(quads=False, nx=N, ny=N)

    # Trial Space: u in H1, q in trace space of Hdiv
    Xo = ng.H1(mesh, order=p+1, dirichlet='.*')
    Xf = ng.HDiv(mesh, order=p, orderinner=0)
    X = Xo * Xf

    # Test Space: error representation function e in broken H1 (represented by L2)
    # Enriched test space order 2 for stability
    Y = ng.L2(mesh, order=p+mesh.dim)

    # Combined Space for Static Condensation
    XY = X * Y
    
    X_pm = NGSolveVectorSpace(X)
    Y_pm = NGSolveVectorSpace(Y)
    
    return mesh, X, Y, XY, X_pm, Y_pm

# =============================================================================
# 2. PARAMETER DEFINITIONS
# =============================================================================
def _define_scenario_params():
    """
    Define parameter regions and RHS.
    """
    print("  > Setting up: 4-Parameter Piecewise Constant Potential")

    # Four quadrants
    pot_regions = [
        RectChar(0.0, 0.5, 0.0, 0.5), RectChar(0.5, 1.0, 0.0, 0.5),
        RectChar(0.0, 0.5, 0.5, 1.0), RectChar(0.5, 1.0, 0.5, 1.0)
    ]
    pot_key = 'blocks'
    ranges = {'blocks': (0.1, 100.0)}
    ng_params = {'blocks': [ng.Parameter(1.0) for _ in range(4)]}

    # Gaussian source (localized at domain center)
    rhs_exprs = [10.0 * exp(-50*((x-0.5)**2 + (y-0.5)**2))]

    # Alternatives: 
    # rhs_exprs = [1.0]  # constant source
    # rhs_exprs = [5.0 * exp(-50*((x-0.25)**2 + (y-0.25)**2)) + 
    #              5.0 * exp(-50*((x-0.75)**2 + (y-0.75)**2))]  # two sources
    
    # source term f is parameter-independent
    rhs_key = None

    return pot_regions, pot_key, rhs_exprs, rhs_key, ranges, ng_params

# =============================================================================
# 3. GRAMIAN ASSEMBLY
# =============================================================================
def _assemble_gramian(Y, Y_pm):
    """
    Assemble test space Gramian G and its inverse.
    """
    e_trial, d_test = Y.TrialFunction(), Y.TestFunction()
    g = ng.BilinearForm(Y, symmetric=True)
    g += (e_trial * d_test + grad(e_trial) * grad(d_test)) * dx
    g.Assemble()
    
    inv_g = g.mat.Inverse(Y.FreeDofs(), inverse="sparsecholesky")
    return NGSolveMatrixOperator(inv_g, Y_pm, Y_pm)

# =============================================================================
# 4. OPERATOR ASSEMBLY (AFFINE DECOMPOSITION) FOR ROM
# =============================================================================
def _assemble_B_operators(X, Y, X_pm, Y_pm, n, u, q, d, pot_regions, pot_key):
    """
    Assemble affine components of the trial-to-test operator B: X -> Y'.
    
    B((u,q), v) = (grad u, grad v) + (V u, v) - <q.n, v>
    Decomposed as B_0 (Laplacian + flux) + sum_i mu_i B_i (potential terms).
    """
    qn = ng.InnerProduct(q, n)
    ops_B = []
    coeffs_B = []
    
    # Parameter-independent part (Laplacian + flux)
    b0 = ng.BilinearForm(trialspace=X, testspace=Y)
    b0 += grad(u) * grad(d) * dx - qn * d * dx(element_boundary=True)
    b0.Assemble()
    ops_B.append(NGSolveMatrixOperator(b0.mat, Y_pm, X_pm))
    coeffs_B.append(1.0)
    
    # Parameter-dependent parts (Potential terms)
    for i, reg in enumerate(pot_regions):
        coef = ProjectionParameterFunctional(pot_key, len(pot_regions), i)
        bi = ng.BilinearForm(trialspace=X, testspace=Y)
        bi += (reg * u * d) * dx
        bi.Assemble()
        ops_B.append(NGSolveMatrixOperator(bi.mat, Y_pm, X_pm))
        coeffs_B.append(coef)
    
    return LincombOperator(ops_B, coeffs_B)


def _assemble_rhs_operators(Y, Y_pm, d, rhs_exprs, rhs_key):
    """
    Assemble affine components of RHS functional f.
    f(mu) = sum_k psi_k(mu) * f_k
    """
    ops_f = []
    coeffs_f = []
    
    for i, expr in enumerate(rhs_exprs):
        F = ng.LinearForm(Y)
        F += expr * d * dx
        F.Assemble()
        
        gf_f = ng.GridFunction(Y)
        gf_f.vec.data = F.vec
        ops_f.append(VectorOperator(Y_pm.make_array([gf_f])))
        
        if rhs_key:
            coeffs_f.append(ProjectionParameterFunctional(rhs_key, len(rhs_exprs), i))
        else:
            coeffs_f.append(1.0)
    
    return LincombOperator(ops_f, coeffs_f), coeffs_f

# =============================================================================
# 5. FOM BILINEAR/LINEAR FORM ASSEMBLY
# =============================================================================
def _assemble_fom_forms(XY, n, pot_regions, pot_key, rhs_exprs, rhs_key, ng_params):
    """
    Assemble NGSolve forms for FOM static condensation solve.
    """
    u_x, q_x, e_y = XY.TrialFunction()
    w_x, r_x, d_y = XY.TestFunction()
    
    a_full = ng.BilinearForm(XY, condense=True)
    f_full = ng.LinearForm(XY)
    
    # Gramian block (test space inner product y(e, d) = (e, d) + (grad e, grad d))
    a_full += (e_y * d_y + grad(e_y) * grad(d_y)) * dx
    
    # b((u,q.n), d): maps trial to test
    a_full += grad(u_x) * grad(d_y) * dx - ng.InnerProduct(q_x, n) * d_y * dx(element_boundary=True)
    # b((w,r.n), e): adjoint block
    a_full += grad(w_x) * grad(e_y) * dx - ng.InnerProduct(r_x,n) * e_y * dx(element_boundary=True)
    
    # Potential terms
    for i, reg in enumerate(pot_regions):
        p_ng = ng_params[pot_key][i]
        a_full += (p_ng * reg * u_x * d_y) * dx
        a_full += (p_ng * reg * w_x * e_y) * dx
    
    # RHS
    for i, expr in enumerate(rhs_exprs):
        if rhs_key:
            p_ng = ng_params[rhs_key][i]
            f_full += p_ng * expr * d_y * dx
        else:
            f_full += expr * d_y * dx
    
    return a_full, f_full


# =============================================================================
# 6. PRODUCT/NORM SETUP
# =============================================================================
def _setup_product(X, X_pm, u, q):
    """
    Setup inner product for basis orthonormalization.
    """
    w, r = X.TestFunction()
    mesh = X.mesh
    n = ng.specialcf.normal(mesh.dim)
    h = ng.specialcf.mesh_size

    norm_form = ng.BilinearForm(X, symmetric=True)

    # Volume H1 product for u
    norm_form += (grad(u) * grad(w) + u * w) * dx
    
    # Pure interface L2 product for the normal trace of q
    norm_form += (h * ng.InnerProduct(q, n) * ng.InnerProduct(r, n)) * dx(element_boundary=True)
    
    norm_form.Assemble()
    
    return NGSolveMatrixOperator(norm_form.mat, X_pm, X_pm)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def setup_scenario(p, maxh):
    """
    Setup complete DPG-RB scenario.
    """    
    # 1. Mesh and Spaces
    mesh, X, Y, XY, X_pm, Y_pm = _create_mesh_and_spaces(maxh, p)
    
    # 2. Scenario parameters
    pot_regions, pot_key, rhs_exprs, rhs_key, ranges, ng_params = _define_scenario_params()
    
    n = ng.specialcf.normal(mesh.dim)
    u, q = X.TrialFunction()
    d = Y.TestFunction()
    
    with ng.TaskManager():
        # Assembly
        op_invG = _assemble_gramian(Y, Y_pm)
        op_B = _assemble_B_operators(X, Y, X_pm, Y_pm, n, u, q, d, pot_regions, pot_key)
        op_f, _ = _assemble_rhs_operators(Y, Y_pm, d, rhs_exprs, rhs_key)
        a_full, f_full = _assemble_fom_forms(XY, n, pot_regions, pot_key, rhs_exprs, rhs_key, ng_params)
        
        # Model Creation
        fom = DPGModel(op_B, op_invG, op_f, X, XY, mesh, ng_params, a_full, f_full)
        product = _setup_product(X, X_pm, u, q)
    
    return fom, X, mesh, ranges, product, pot_regions, pot_key