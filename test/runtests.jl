using FieldDimArrays
using FieldDimArrays: ncomponents, fieldshape, componenttype, componentnames, fromcomponents,
    components, withcomponenttype, isfieldelement
using Test
using JET
using StaticArrays
using Adapt

struct Point3D
    x::Float64
    y::Float64
    z::Float64
end

struct Point2D{T}
    x::T
    y::T
end

struct RGBA
    r::Float32
    g::Float32
    b::Float32
    a::Float32
end

struct Mixed
    a::Float64
    b::Int
end

struct Stokes{T} <: FieldVector{4, T}
    I::T
    Q::T
    U::T
    V::T
end

@testset "FieldDimArrays.jl" begin
    include("core.jl")
    include("staticarrays.jl")
    include("broadcast.jl")
    include("reactant.jl")
end
