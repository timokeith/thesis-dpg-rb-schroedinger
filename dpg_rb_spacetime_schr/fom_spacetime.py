"""
DPG Full Order Model wrapper for pyMOR, 
including short FOM Error Estimator class computing norm of error representation function, 
i.e. residual norm wrt mean parameter in enriched test space Y (the usual DPG a posteriori error estimator)

Uses NGSolve static condensation for the FOM solve, and provides
affinely decomposed normal equation operators for ROM construction.
"""

import numpy as np
import ngsolve as ng

from pymor.core.base import ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, VectorOperator

from utils_spacetime import Adj_A, reconstruct_potential

# =============================================================================
# 1. CUSTOM FOM DPG MODEL
# =============================================================================
class DPGModel(StationaryModel):
    """DPG model combining NGSolve FOM solver with pyMOR operator structure."""
    _parameters_varargs_warning = False
    def __init__(self, op_B, op_G_inv, op_f, X_space, XY_space, mesh, 
                 ng_params, a_full, f_full, pot_decomp=None, horiz_qprime_dofs=None, **kwargs):
        # Store arguments as attributes for pyMOR to handle them correctly
        self.op_B = op_B
        self.op_G_inv = op_G_inv
        self.op_f = op_f 
        self.X_space = X_space
        self.XY_space = XY_space
        self.mesh = mesh
        self.pot_decomp = pot_decomp
        self.horiz_qprime_dofs = horiz_qprime_dofs or set()
        
        # Static Condensation Objects
        self.ng_params = ng_params
        self.a_full = a_full
        self.f_full = f_full

        # Reusable buffer for FOM solve (avoids repeated allocation)
        self._gfu_buffer = ng.GridFunction(self.XY_space)

        # Assemble the Reduced Operators (Affine Decomposition for ROM)
        op_LHS_NE, op_RHS_NE = self._assemble_normal_equations()

        # Initialize custom FOM error estimator (see below)
        self.error_estimator = FOMErrorEstimator(mesh, pot_decomp)
        
        super().__init__(op_LHS_NE, op_RHS_NE, **kwargs)

    def _assemble_normal_equations(self):
        ops_b = self.op_B.operators
        coeffs_b = self.op_B.coefficients
        ops_f = self.op_f.operators
        coeffs_f = self.op_f.coefficients
        
        # LHS Assembly: B_i^T G^-1 B_j
        lhs_ops_NE = []
        lhs_coeffs_NE = []
        
        print("  > Assembling LHS components...")
        for i, Bi in enumerate(ops_b):
            for j in range(i, len(ops_b)):
                Bj = ops_b[j]
                # Operator composition handled by pyMOR
                op_chain = Bi.H @ self.op_G_inv @ Bj
                lhs_ops_NE.append(op_chain)

                coeff = coeffs_b[i] * coeffs_b[j] # note that coefficients are purely real
                lhs_coeffs_NE.append(coeff) 

                if i != j:
                    lhs_ops_NE.append(Bj.H @ self.op_G_inv @ Bi)
                    lhs_coeffs_NE.append(coeff)
                    
        op_LHS_NE = LincombOperator(lhs_ops_NE, lhs_coeffs_NE)

        # RHS Assembly: B_i^T G^-1 f_k
        print("  > Assembling RHS components...")
        rhs_ops_NE = []
        rhs_coeffs_NE = []

        for k, fk in enumerate(ops_f):
            # Apply G^-1 to f_k once
            ginv_fk = self.op_G_inv.apply(fk.as_range_array())
            for i, Bi in enumerate(ops_b):
                v_vec = Bi.H.apply(ginv_fk)
                rhs_ops_NE.append(VectorOperator(v_vec))
                rhs_coeffs_NE.append(coeffs_b[i] * coeffs_f[k])

        op_RHS_NE = LincombOperator(rhs_ops_NE, rhs_coeffs_NE)
        return op_LHS_NE, op_RHS_NE

    def _compute(self, quantities, data, mu=None):
        """
        FOM Solve using NGSolve Static Condensation, 
        called internally by solve-method.
        """

        # We need to solve before computing the error estimate since the error
        # representation e is obtained from the mixed system solve (not stored in solution).
        if 'solution' in quantities or 'solution_error_estimate' in quantities:
            # Update parameters
            for name, val in mu.items():
                if name in self.ng_params:
                    for p_ng in self.ng_params[name]:
                            p_ng.Set(val[0])

            # Re-Assemble and solve
            with ng.TaskManager():    
                self.a_full.Assemble()
                self.f_full.Assemble()
                
                self._gfu_buffer.vec[:] = 0.0 + 0.0j
                f_vec = self.f_full.vec
                
                f_schur = f_vec.CreateVector()
                f_schur.data = f_vec + self.a_full.harmonic_extension_trans * f_vec
                
                coupling_freedofs = ng.BitArray(self.XY_space.ndof)
                coupling_freedofs[:] = self.XY_space.FreeDofs(coupling=True)
                for d in self.horiz_qprime_dofs:
                    coupling_freedofs.Clear(d)

                inv = self.a_full.mat.Inverse(coupling_freedofs, inverse="pardiso")
                self._gfu_buffer.vec.data = inv * f_schur
                self._gfu_buffer.vec.data += self.a_full.harmonic_extension * self._gfu_buffer.vec
                self._gfu_buffer.vec.data += self.a_full.inner_solve * f_vec
                
                sol_gf = ng.GridFunction(self.X_space) 
                sol_gf.components[0].vec.data = self._gfu_buffer.components[0].vec
                sol_gf.components[1].vec.data = self._gfu_buffer.components[1].vec
                sol_gf.components[2].vec.data = self._gfu_buffer.components[2].vec

                data['solution'] = self.operator.source.make_array([sol_gf])
                quantities.remove('solution')
            
        # Compute FOM DPG Residual ||e||_Y
        if 'solution_error_estimate' in quantities:
            error_repr = self._gfu_buffer.components[3]
            data['solution_error_estimate'] = self.error_estimator.estimate_error(error_repr, mu, self)
            quantities.remove('solution_error_estimate')

        # Just of formal interest here
        super()._compute(quantities, data, mu=mu)

# =============================================================================
# 2. CUSTOM FOM ERROR ESTIMATOR
# =============================================================================
class FOMErrorEstimator(ImmutableObject):
    """Computes the exact residual norm in the enriched test space Y."""
    def __init__(self, mesh, pot_decomp):
        self.__auto_init(locals())

    def estimate_error(self, error_repr, mu, fom):
        V_mu = reconstruct_potential(mu, self.pot_decomp)
            
        # Compute Norm: ||e||_Y^2 = ||e||^2 + ||A*(mu) e||^2
        val_sq = ng.Integrate(error_repr * ng.Conj(error_repr) + 
            Adj_A(error_repr, V_mu) * ng.Conj(Adj_A(error_repr, V_mu)), 
            self.mesh, order=25).real
        
        return np.array([np.sqrt(val_sq)])
