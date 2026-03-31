"""
DPG-specific ROM components: error estimator and reductor.
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
    Computes dual norm of the residual || f - B u_N ||_{Y'} via naive affine expansion
    (||.||_{Y'} is induced by G^{-1}).
    Expanded: f^T G^{-1} f  -  2 u_N^T (B^T G^{-1} f)  +  u_N^T (B^T G^{-1} B) u_N
    This class enables offline/online decomposition of these terms.
    """

    def __init__(self, op_f, op_G_inv):
        """
        Parameters:
            op_f (LincombOperator): Affine decomposition of the functional f.
            op_G_inv (Operator): Inverse Gramian operator G^{-1}.
        """
        self.__auto_init(locals())
        
        self._f_ops = self.op_f.operators
        self._f_coeffs = self.op_f.coefficients
        n_ops = len(self._f_ops)

        # OFFLINE: Precompute C_ff matrix (f_m^T G^{-1} f_n) to avoid 
        # full size online operations when computing ||f(mu)||^2
        # (this approach can be as well be used for a parameter-dependent RHS).
        print("  > Initializing Error Estimator components...")
        f_vecs = [fop.as_range_array() for fop in self._f_ops]
        ginv_f_vecs = [self.op_G_inv.apply(fv) for fv in f_vecs]

        self.Cff = np.zeros((n_ops, n_ops))
        for m in range(n_ops):
            for n in range(m,n_ops):
                val = f_vecs[m].inner(ginv_f_vecs[n])[0, 0]
                self.Cff[m, n] = val
                self.Cff[n,m] = val

    def estimate_error(self, U, mu, rom):
        """
        Evaluate ||f(mu) - B(mu) U||_{G^{-1}}.
        
        Uses the Normal Equation components rom.operator and rom.rhs for the reduced linear/quadratic terms,
        and precomputed Cff for the constant term.
        """
        # 1. Quadratic Term: u_N^T A_NE(mu) u_N
        AU = rom.operator.apply(U, mu)
        quad = U.inner(AU)[0, 0]

        # 2. Linear Term: u_N^T F_NE(mu)
        F = rom.rhs.as_range_array(mu)
        lin = U.inner(F)[0, 0]

        # 3. Constant Term: psi(mu)^T C_ff psi(mu)
        psi = np.array([c(mu) if callable(c) else float(c) for c in self._f_coeffs], dtype=float)
        const = float(psi @ (self.Cff @ psi))

        # 4. Combine: ||e||^2 = const + quad - 2*lin
        err_sq = const + quad - 2.0 * lin
        
        # Numerical Safety Check
        if err_sq < 0:
            if abs(err_sq) < 1e-10 * max(const, quad):
                # Within numerical tolerance - treat as zero
                return np.array([0.0])
            else:
                print(f" >> WARNING: Estimated error {err_sq:.2e} is significantly negative!")
                return np.array([0.0])
            
        return np.array([np.sqrt(err_sq)])
    
# =============================================================================
# 2. PROJECTED REDUCTOR
# =============================================================================
class DPGProjectedReductor(StationaryRBReductor):
    """
    Galerkin projection reductor with DPG error estimator.
    """
    def __init__(self, fom, RB=None, product=None, 
                check_orthonormality=None, check_tol=None):
        
        super().__init__(fom, RB=RB, product=product, 
                        check_orthonormality=check_orthonormality, check_tol=check_tol)

        # Store references for error estimator construction
        self._op_f = fom.op_f
        self._op_G_inv = fom.op_G_inv
        self.error_estimator = None

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
        if 'L2' in self.fom.products:
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
        if 'L2' in self._last_rom.products:
            projected_ops['products']['L2'] = project_to_subbasis(self._last_rom.products['L2'], dim, dim)
            
        if self._last_rom.output_functional is not None:
            projected_ops['output_functional'] = project_to_subbasis(self._last_rom.output_functional, None, dim)
        return projected_ops

    def assemble_error_estimator(self):
        """Called by reduce() to attach error estimator to ROM."""
        if self.error_estimator is None:
            self.error_estimator = ROMErrorEstimator(self._op_f, self._op_G_inv)
        return self.error_estimator
    
    def assemble_error_estimator_for_subbasis(self, dims):
        """Called by reduce() for subbasis error estimators (same estimator works)."""
        return self.assemble_error_estimator()