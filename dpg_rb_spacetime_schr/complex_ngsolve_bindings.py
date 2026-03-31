"""
Complex NGSolve Bindings to pyMOR correctly handling and storing complex values
Bypasses pyMOR's LinearComplexifiedListVectorArrayOperatorBase and NGSolveVectorSpace entirely.
Allows for efficient complex operations without real/imag splits.
"""

import numpy as np
import ngsolve as ng
from pymor.vectorarrays.list import ListVectorSpace
from pymor.bindings.ngsolve import NGSolveVector
from pymor.operators.list import ListVectorArrayOperatorBase

class ComplexNGSolveVector(NGSolveVector):
    """Complex NGSolve vector implementation."""
    
    def _scal(self, alpha):
        self.impl.vec.data = complex(alpha) * self.impl.vec

    def _axpy(self, alpha, x):
        self.impl.vec.data = self.impl.vec + complex(alpha) * x.impl.vec

    def inner(self, other):
        return other.impl.vec.InnerProduct(self.impl.vec)
    
    def norm2(self):
        return self.impl.vec.InnerProduct(self.impl.vec).real

    def norm(self):
        return np.sqrt(self.norm2())

class ComplexNGSolveVectorSpace(ListVectorSpace):
    """Complex VectorSpace for NGSolve objects."""
    def __init__(self, V):
        self.__auto_init(locals())

    def __eq__(self, other):
        return type(other) is ComplexNGSolveVectorSpace and self.V == other.V

    def __hash__(self):
        return hash(self.V)

    @property
    def value_dim(self):
        u = self.V.TrialFunction()
        return u[0].dim if isinstance(u, list) else u.dim

    @property
    def dim(self):
        return self.V.ndofglobal * self.value_dim

    @classmethod
    def space_from_vector_obj(cls, vec):
        return cls(vec.space)

    def zero_vector(self):
        impl = ng.GridFunction(self.V)
        return ComplexNGSolveVector(impl)

    def make_vector(self, obj):
        return ComplexNGSolveVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        v = self.zero_vector()
        v.to_numpy()[:] = data
        return v

class ComplexNGSolveMatrixOperator(ListVectorArrayOperatorBase):
    """Matrix operator for complex NGSolve matrices."""
    
    linear = True

    def __init__(self, matrix, range, source, solver_options=None, name=None):
        self.__auto_init(locals())

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        res = self.range.zero_vector()
        self.matrix.Mult(u.impl.vec, res.impl.vec)
        return res

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        res = self.source.zero_vector()
        
        v_conj = v.impl.vec.CreateVector()
        v_conj.FV().NumPy()[:] = v.impl.vec.FV().NumPy().conj()

        self.matrix.T.Mult(v_conj, res.impl.vec)
        res.impl.vec.FV().NumPy()[:] = res.impl.vec.FV().NumPy().conj()
        
        return res

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None, **kwargs):
        if not all(isinstance(op, ComplexNGSolveMatrixOperator) for op in operators):
            return NotImplemented
        if identity_shift != 0:
            return NotImplemented

        matrix = operators[0].matrix.CreateMatrix()
        matrix.AsVector().data = complex(coefficients[0]) * operators[0].matrix.AsVector()
        
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.AsVector().data += complex(c) * op.matrix.AsVector()
            
        return ComplexNGSolveMatrixOperator(matrix, self.range, self.source, solver_options=solver_options, name=name)