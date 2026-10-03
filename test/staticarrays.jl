@testset "StaticArrays elements" begin
    @testset "SVector" begin
        P = reshape(collect(1.0:12.0), 4, 3)
        a = @inferred ViewStructArray{SVector{3}}(P)
        @test a isa ViewStructArray{SVector{3, Float64}, 1, Matrix{Float64}}
        @test a[2] === SVector(2.0, 6.0, 10.0)
        a[2] = SVector(0.0, 1.0, 2.0)
        @test P[2, :] == [0.0, 1.0, 2.0]
        @test propertynames(a) == ()
        @test fieldview(a, 3) == P[:, 3]
        @test_throws "has no component named :x" a.x
        @test ncomponents(SVector{3, Float64}) == 3
        @test fieldshape(SVector{3, Float64}) == (3,)
        @test isviewelement(SVector{3, Float64})
        @test !isviewelement(MVector{3, Float64})
        @test !isviewelement(SVector{3})
        @test withcomponenttype(SVector{3}, Float32) == SVector{3, Float32}
    end

    @testset "SMatrix{2, 2} over (..., 2, 2)" begin
        P = reshape(collect(1.0:24.0), 3, 2, 2, 2)
        a = @inferred ViewStructArray{SMatrix{2, 2}}(P)
        @test a isa ViewStructArray{SMatrix{2, 2, Float64, 4}, 2, Array{Float64, 4}}
        @test size(a) == (3, 2)
        @test a[2, 1] === SMatrix{2, 2}(P[2, 1, 1, 1], P[2, 1, 2, 1], P[2, 1, 1, 2], P[2, 1, 2, 2])
        @test a[2, 1] == P[2, 1, :, :]
        @test a[5] == a[2, 2]
        a[1, 2] = SMatrix{2, 2}(1.0, 2.0, 3.0, 4.0)
        @test P[1, 2, :, :] == [1.0 3.0; 2.0 4.0]
        @test fieldview(a, 2) == P[:, :, 2, 1]
        @test fieldview(a, 1, 2) == P[:, :, 1, 2]
        @test fieldview(a, 2, 1) == fieldview(a, 2)
        @test_throws "component indices (3, 1) do not index the trailing size (2, 2)" fieldview(a, 3, 1)
        @test fieldshape(SMatrix{2, 2, Float64, 4}) == (2, 2)

        flat = @inferred ViewStructArray{SMatrix{2, 2}, 1}(reshape(P, 3, 2, 4)[:, 1, :])
        @test size(flat) == (3,)
        @test flat[2] == a[2, 1]
        @test_throws "the trailing size of the parent must be (2, 2)" ViewStructArray{SMatrix{2, 2}}(rand(3, 2, 3))
        @test_throws "needs a parent with 3 dims (trailing size (2, 2)) or 2 dims (trailing size (4,)); got 4 dims" ViewStructArray{SMatrix{2, 2}, 1}(rand(3, 2, 2, 2))

        s = @inferred view(a, 2:3, 1)
        @test s isa ViewStructArray{SMatrix{2, 2, Float64, 4}, 1}
        @test s == [a[2, 1], a[3, 1]]
        @test @inferred(similar(a)) isa ViewStructArray{SMatrix{2, 2, Float64, 4}, 2, Array{Float64, 4}}
        @test size(parent(similar(flat))) == (3, 4)
        @test size(parent(similar(flat, SMatrix{2, 2, Float64, 4}, (5,)))) == (5, 4)
        @test size(parent(similar(a, SMatrix{2, 2, Float32, 4}, (5,)))) == (5, 2, 2)

        read_cartesian(a, 2, 1); write_cartesian!(a, a[1], 2, 1); read_linear(a, 4); write_linear!(a, a[1], 4)
        @test (@allocated read_cartesian(a, 2, 1)) == 0
        @test (@allocated read_linear(a, 4)) == 0
        @test (@allocated write_cartesian!(a, a[1], 2, 1)) == 0
        @test (@allocated write_linear!(a, a[1], 4)) == 0
        @test @inferred(read_linear(a, 4)) isa SMatrix{2, 2, Float64, 4}
        JET.@test_opt target_modules = (ViewStructArrays,) read_cartesian(a, 2, 1)
        JET.@test_opt target_modules = (ViewStructArrays,) write_linear!(a, a[1], 4)
    end

    @testset "FieldVector" begin
        P = rand(5, 4)
        a = @inferred ViewStructArray{Stokes}(P)
        @test a isa ViewStructArray{Stokes{Float64}, 1, Matrix{Float64}}
        @test a[3] === Stokes(P[3, 1], P[3, 2], P[3, 3], P[3, 4])
        @test propertynames(a) == (:I, :Q, :U, :V)
        @test @inferred(getproperty(a, :U)) == P[:, 3]
        @test a.V == fieldview(a, 4)
        @test withcomponenttype(Stokes{Float64}, Float32) == Stokes{Float32}
        @test withcomponenttype(Stokes, Float64) == Stokes{Float64}
        @test isviewelement(Stokes{Float64})
        a[1] = Stokes(1.0, 2.0, 3.0, 4.0)
        @test P[1, :] == [1.0, 2.0, 3.0, 4.0]
        @test @inferred(elementview(Stokes, P)) == a
    end

    @testset "complex static components" begin
        P = rand(ComplexF64, 3, 2)
        a = ViewStructArray{SVector{2}}(P)
        @test eltype(a) == SVector{2, ComplexF64}
        @test a[2] == SVector(P[2, 1], P[2, 2])
    end
end
