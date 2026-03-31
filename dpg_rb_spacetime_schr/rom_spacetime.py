"""
DPG-specific ROM components: error estimator and reductor, using normal equations and affine error estimator expansion
"""

import numpy as np
from pymor.core.base import ImmutableObject
from pymor.reductors.basic import StationaryRBReductor
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

# =============================================================================
# 1. CUSTOM ROM ERROR ESTIMATOR
# =============================================================================
class ROMErrorEstimator(ImmutableObject):
    """
    Computes the dual norm of the residual, using naive affine expansion.
    """

    def __init__(self, op_f, op_G_inv, const_ff = None, use_l2_product=False):
        self.__auto_init(locals())
        
        self._f_ops = self.op_f.operators
        self._f_coeffs = self.op_f.coefficients
        
        if self.const_ff is None:
            f_vec = self.op_f.operators[0].as_range_array()
            ginv_f = self.op_G_inv.apply(f_vec)
            self.const_ff = f_vec.inner(ginv_f)[0, 0].real
            print(f"  > Error Estimator: ||f||²_{{G^{{-1}}}} = {self.const_ff:.6e}")

    def estimate_error(self, U, mu, rom):
        """
        Evaluate ||f(mu) - B(mu) U||_{G_{mu_mean}^{-1}} using reduced normal equation components.
        """

        # Quadratic Term
        AU = rom.operator.apply(U, mu)
        quad = U.inner(AU)[0, 0].real

        # Linear Term
        F = rom.rhs.as_range_array(mu)
        lin = U.inner(F)[0, 0].real

        # ||r||²_{G^{-1}} = ||f||²_{G^{-1}} + u_N^H A_NE u_N - 2 Re(u_N^H F_NE)
        resid_norm_sq = self.const_ff + quad - 2.0 * lin
        
        # Handle numerical zero
        if resid_norm_sq < 0:
            if abs(resid_norm_sq) < 1e-9 * max(abs(self.const_ff), abs(quad), 1e-15):
                # Within numerical tolerance - treat as zero
                resid_norm_sq = 0.0
            else:
                print(f" >> WARNING: Resid^2 = {resid_norm_sq:.4e} is significantly negative!"
                      f"\n(const={self.const_ff:.4e}, quad={quad:.4e}, lin={lin:.4e})")
                resid_norm_sq = 0.0
        
        resid_norm = np.sqrt(resid_norm_sq)
        
        return np.array([resid_norm])
    
# =============================================================================
# 2. PROJECTED REDUCTOR
# =============================================================================
class DPGProjectedReductor(StationaryRBReductor):
    """
    Galerkin projection reductor with DPG error estimator.
    """
    def __init__(self, fom, RB=None, product=None, 
                check_orthonormality=None, check_tol=None, use_l2_product=False):
        
        super().__init__(fom, RB=RB, product=product, 
                         check_orthonormality=check_orthonormality, check_tol=check_tol)

        # Store references for error estimator construction
        self._op_f = fom.op_f
        self._op_G_inv = fom.op_G_inv
        self.error_estimator = None
        self.use_l2_product = use_l2_product

        # Cache applied operators to avoid repeated solves during projection
        self._b_applied = [fom.op_G_inv.source.empty() for _ in fom.op_B.operators]
        self._supremizers = [fom.op_G_inv.range.empty() for _ in fom.op_B.operators]
        self._ginv_f = [fom.op_G_inv.apply(f.as_range_array()) for f in fom.op_f.operators]

    def extend_basis(self, U, **kwargs):
        old_size = len(self.bases['RB'])
        super().extend_basis(U, **kwargs)
        U_new = self.bases['RB'][old_size:]
        
        for q, B_q in enumerate(self.fom.op_B.operators):
            b_q_U = B_q.apply(U_new)
            w_q = self.fom.op_G_inv.apply(b_q_U)
            self._b_applied[q].append(b_q_U)
            self._supremizers[q].append(w_q)

    def project_operators(self):
        red_space = NumpyVectorSpace(len(self.bases['RB']))
        ops_b, coeffs_b = self.fom.op_B.operators, self.fom.op_B.coefficients
        
        lhs_ops, lhs_coeffs = [], []
        for i in range(len(ops_b)):
            for j in range(i, len(ops_b)):
                mat = self._b_applied[i].inner(self._supremizers[j])
                lhs_ops.append(NumpyMatrixOperator(mat))
                lhs_coeffs.append(coeffs_b[i] * coeffs_b[j])
                if i != j:
                    mat_sym = self._b_applied[j].inner(self._supremizers[i])
                    lhs_ops.append(NumpyMatrixOperator(mat_sym))
                    lhs_coeffs.append(coeffs_b[i] * coeffs_b[j])
        rom_op = LincombOperator(lhs_ops, lhs_coeffs)

        rhs_ops, rhs_coeffs = [], []
        coeffs_f = self.fom.op_f.coefficients
        for k in range(len(self.fom.op_f.operators)):
            for i in range(len(ops_b)):
                vec = self._b_applied[i].inner(self._ginv_f[k]).flatten()
                rhs_ops.append(VectorOperator(red_space.make_array(vec)))
                rhs_coeffs.append(coeffs_b[i] * coeffs_f[k])
        rom_rhs = LincombOperator(rhs_ops, rhs_coeffs)

        projected_ops = {'operator': rom_op, 'rhs': rom_rhs, 'products': {}}
        if not self.use_l2_product and 'L2' in self.fom.products:
            projected_ops['products']['L2'] = project(self.fom.products['L2'], self.bases['RB'], self.bases['RB'])
            
        if self.fom.output_functional is not None:
            projected_ops['output_functional'] = project(self.fom.output_functional, None, self.bases['RB'])
        return projected_ops

    def project_operators_to_subbasis(self, dims):
        dim = dims['RB']
        projected_ops = {
            'operator': project_to_subbasis(self._last_rom.operator, dim, dim),
            'rhs': project_to_subbasis(self._last_rom.rhs, dim, None),
            'products': {}
        }
        if not self.use_l2_product and 'L2' in self._last_rom.products:
            projected_ops['products']['L2'] = project_to_subbasis(self._last_rom.products['L2'], dim, dim)
            
        if self._last_rom.output_functional is not None:
            projected_ops['output_functional'] = project_to_subbasis(self._last_rom.output_functional, None, dim)
        return projected_ops
    
    def assemble_error_estimator(self):
        """Called by reduce() to attach error estimator to ROM."""
        if self.error_estimator is None:
            self.error_estimator = ROMErrorEstimator(self._op_f, self._op_G_inv, use_l2_product = self.use_l2_product)
        else:
            self.error_estimator = self.error_estimator.with_(use_l2_product=self.use_l2_product, const_ff=self.error_estimator.const_ff)
        return self.error_estimator
    
    def assemble_error_estimator_for_subbasis(self, dims):
        """Called by reduce() for subbasis error estimators (same estimator works)."""
        return self.assemble_error_estimator()