"""
DPG Full Order Model wrapper for pyMOR, 
including short FOM Error Estimator class computing norm of error representation function, 
i.e. residual norm in enriched test space Y (the usual DPG a posteriori error estimator)

Uses NGSolve static condensation for the FOM solve, and provides
affinely decomposed normal equation operators for ROM construction.
"""

import numpy as np
import ngsolve as ng
from ngsolve import grad

from pymor.core.base import ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import LincombOperator, VectorOperator

# =============================================================================
# 1. CUSTOM FOM DPG MODEL
# =============================================================================
class DPGModel(StationaryModel):
    """
    DPG model combining NGSolve FOM solver with pyMOR operator structure.
    
    The solve() method uses static condensation; the inherited operator/rhs
    attributes contain the normal equation system for Galerkin projection.
    """
    def __init__(self, op_B, op_G_inv, op_f, X_space, XY_space, mesh, 
                 ng_params, fom_bform, fom_lform):
        """
        Parameters:
            op_B, op_G_inv, op_f : pyMOR operators (for ROM normal equations)
            X_space, XY_space : NGSolve FESpaces
            mesh : NGSolve Mesh
            ng_params : dict of NGSolve Parameter lists (for mu updates)
            fom_bform, fom_lform : NGSolve forms (condense=True) for FOM solve
        """
        # Store arguments as attributes for pyMOR being able to handle them correctly
        self.op_B = op_B
        self.op_G_inv = op_G_inv
        self.op_f = op_f 
        self.X_space = X_space
        self.XY_space = XY_space
        self.mesh = mesh
        
        # Static Condensation Objects
        self.ng_params = ng_params
        self.fom_bform = fom_bform
        self.fom_lform = fom_lform

        # Reusable buffer for FOM solve (avoids repeated allocation)
        self._gfu_buffer = ng.GridFunction(self.XY_space)

        # Assemble the Reduced Operators (Affine Decomposition for ROM)
        op_LHS_NE, op_RHS_NE = self._assemble_normal_equations()

        # Initialize custom FOM error estimator (see below)
        self.error_estimator = FOMErrorEstimator(mesh)
        
        super().__init__(op_LHS_NE, op_RHS_NE)

    def _assemble_normal_equations(self):
        """
        Assembles affine components of the DPG normal equations (for ROM).
        
        For B(mu) = sum_i theta_i(mu) B_i and f(mu) = sum_k psi_k(mu) f_k:
          LHS_ij = B_i^T G^{-1} B_j  (combined with theta_i * theta_j)
          RHS_ik = B_i^T G^{-1} f_k  (combined with theta_i * psi_k)
        """
        ops_b = self.op_B.operators
        coeffs_b = self.op_B.coefficients
        ops_f = self.op_f.operators
        coeffs_f = self.op_f.coefficients
        
        # 1. LHS Assembly: B_i^T G^-1 B_j
        lhs_ops_NE = []
        lhs_coeffs_NE = []
        
        print("  > Assembling LHS components...")
        for i, Bi in enumerate(ops_b):
            for j in range(i, len(ops_b)):
                Bj = ops_b[j]
                # Operator composition handled by pyMOR
                op_chain = Bi.H @ self.op_G_inv @ Bj
                lhs_ops_NE.append(op_chain)

                coeff = coeffs_b[i] * coeffs_b[j]
                lhs_coeffs_NE.append(coeff)

                if i != j:
                    lhs_ops_NE.append(Bj.H @ self.op_G_inv @ Bi)
                    lhs_coeffs_NE.append(coeff)
                    
        op_LHS_NE = LincombOperator(lhs_ops_NE, lhs_coeffs_NE)

        # 2. RHS Assembly: B_i^T G^-1 f_k
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
            # 1. Update NGSolve Parameters
            for name, val in mu.items():
                if name in self.ng_params:
                    param_objs = self.ng_params[name]
                    for p_ng, v_val in zip(param_objs, val):
                            p_ng.Set(v_val)

            # 2. Re-Assemble Forms
            with ng.TaskManager():    
                self.fom_bform.Assemble()
                self.fom_lform.Assemble()
                
                # 3. Static Condensation Solve Sequence
                self._gfu_buffer.vec[:] = 0.0
                f_vec = self.fom_lform.vec
                
                # 3a. Compute modified RHS for Schur complement: (I + H^T) f
                f_schur = f_vec.CreateVector()
                f_schur.data = f_vec + self.fom_bform.harmonic_extension_trans * f_vec
                
                # 3b. Solve Schur Complement (Interface System)
                inv = self.fom_bform.mat.Inverse(self.XY_space.FreeDofs(coupling=True), 
                                                inverse="sparsecholesky")
                self._gfu_buffer.vec.data = inv * f_schur
                
                # 3c. Recover Internal DOFs via harmonic extension
                self._gfu_buffer.vec.data += self.fom_bform.harmonic_extension * self._gfu_buffer.vec
                
                # 3d. Add contribution from internal load (uses original f)
                self._gfu_buffer.vec.data += self.fom_bform.inner_solve * f_vec

                # 4. Extract Solution (u,q) into pyMOR VectorArray
                sol_gf = ng.GridFunction(self.X_space) 
                sol_gf.components[0].vec.data = self._gfu_buffer.components[0].vec
                sol_gf.components[1].vec.data = self._gfu_buffer.components[1].vec

                data['solution'] = self.operator.source.make_array([sol_gf])
                quantities.remove('solution')
            
        # 5. Optional: Compute FOM DPG Residual ||e||_Y
        if 'solution_error_estimate' in quantities:
            # Extract error representation component of mixed system solution
            error_repr = self._gfu_buffer.components[2]

            # Save error estimator value in data dictionary.
            # Note: we pass error_repr (not data['solution']) since e is not part of 
            # the returned solution (u,q). We still use a separate estimator class
            # for consistency with pyMOR's interface.
            data['solution_error_estimate'] = self.error_estimator.estimate_error(error_repr, mu, self)
            quantities.remove('solution_error_estimate')

        # Just of formal interest here
        super()._compute(quantities, data, mu=mu)

# =============================================================================
# 2. CUSTOM FOM ERROR ESTIMATOR
# =============================================================================
class FOMErrorEstimator(ImmutableObject):
    """
    Calculate Y-norm of error representation function, 
    equaling the residual norm in the enriched test space Y
    """
    def __init__(self, mesh):
        self.__auto_init(locals())

    # This structure is required by pymor (even though mu and fom are not required here)
    def estimate_error(self, error_repr, mu, fom):
        # Test space norm (broken H1)
        fom_residual = np.sqrt(ng.Integrate(error_repr**2 + grad(error_repr)**2, self.mesh))
        return np.array([fom_residual])
