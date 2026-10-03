module ViewStructArraysReactantExt

using Reactant: Reactant, AnyTracedRArray
using Reactant.TracedRArrayOverrides: AbstractReactantArrayStyle
using ViewStructArrays: ViewStructArrays, ViewStructArray, ViewStructArrayStyle, _withcomponents

ViewStructArrays.componentwise(::AnyTracedRArray) = true

function Base.BroadcastStyle(::ViewStructArrayStyle{N}, ::AbstractReactantArrayStyle{M}) where {N, M}
    return ViewStructArrayStyle{max(N, M)}()
end

# The element type of a traced `ViewStructArray` follows the element type of its traced parent.
Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(V::Type{<:ViewStructArray}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime)
    )
    V isa DataType || return V
    T, N, A = V.parameters
    A2 = Reactant.traced_type_inner(A, seen, mode, track_numbers, ndevices, runtime)
    return ViewStructArray{_withcomponents(T, eltype(A2)), N, A2}
end

end
