module ViewStructArraysAdaptExt

using Adapt: Adapt
using ViewStructArrays: ViewStructArray, _rewrap

Adapt.adapt_structure(to, a::ViewStructArray) = _rewrap(a, Adapt.adapt(to, parent(a)))
Adapt.parent_type(::Type{<:ViewStructArray{T, N, A}}) where {T, N, A} = A

end
