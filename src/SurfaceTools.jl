module SurfaceTools

using LinearAlgebra
#using ForwardDiff
using StaticArrays
using ApproxFun
using Calculus

import Base: getindex
export ApproxFunCompatible, getindex

export tangentials, normal, gram_det, curvature

struct ApproxFunCompatible{F, G, H} <: Function
    fs::Tuple{F, G, H}
end

function (f::ApproxFunCompatible{F, G, H})(p::T) where {F, G, H, T}
    return SVector(Tuple(f.fs[i](p) for i in 1:3))
end

getindex(f::A, i::Int) where A <: ApproxFunCompatible = f.fs[i]

function tangentials(f::F, p) where F <: Union{Fun, ApproxFunCompatible}
    J = @SMatrix [(Derivative(space(f[j]), i == 1 ? [1, 0] : [0, 1]) * f[j])(p)
                  for j in 1:3, i in 1:2]
    return J[:, 1], J[:, 2]
end

function tangentials(f, p)
    J = Calculus.jacobian(f, Vector(p), :central)
    return J[:, 1], J[:, 2]
end

function normal(f, p)
    t1, t2 = tangentials(f, p)
    return t1 × t2
end

gram_det(f, p) = norm(normal(f, p))

function _first_form(f, p)
    t1, t2 = tangentials(f, p)
    # we use StaticArrays to get fast inversion of 2x2 matrices
    return @SMatrix [t1 ⋅ t1  t1 ⋅ t2;
                     t1 ⋅ t2  t2 ⋅ t2]
end

function _hessian(f::F, p) where F <: Union{Fun, ApproxFunCompatible}
    return Tuple(@SMatrix [(Derivative(space(f[k]),
                        ifelse(i == 1, [1, 0], [0, 1]) .+
                        ifelse(j == 1, [1, 0], [0, 1])) * f[k])(p)
                  for i in 1:2, j in 1:2] for k in 1:3)
end

function _hessian(f, p)
    return Calculus.hessian(p -> f(p)[1], p),
           Calculus.hessian(p -> f(p)[2], p),
           Calculus.hessian(p -> f(p)[3], p)
end

_dot(a, b) = reduce(+, map(*, a, b))

function _second_form(f, p)
    hessian = _hessian(f, p)
    nv = normalize(normal(f, p))
    return _dot(hessian, nv)
end

"""
    curvature(f, p)

    Curvature of a surface parameterized by `f` at parameters `p`.
    Note: Gives curvature w.r.t. the normal vector; may be negative.
"""
function curvature(f, p)
    form1 = _first_form(f, p)
    form2 = _second_form(f, p)
    return 0.5(vec(inv(form1)) ⋅ vec(form2))
end

end # module
