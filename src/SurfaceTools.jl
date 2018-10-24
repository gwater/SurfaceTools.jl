module SurfaceTools

using LinearAlgebra
using ForwardDiff
using StaticArrays

export tangentials, normal, gram_det, curvature

function tangentials(f, p)
    J = ForwardDiff.jacobian(f, p)
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
    return SMatrix{2,2}(t1 ⋅ t1, t1 ⋅ t2,
                        t1 ⋅ t2, t2 ⋅ t2)
end

function _hessian(f, p)
    return ForwardDiff.hessian(p -> f(p)[1], p),
           ForwardDiff.hessian(p -> f(p)[2], p),
           ForwardDiff.hessian(p -> f(p)[3], p)
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
