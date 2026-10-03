read_cartesian(a, i, j) = a[i, j]
read_linear(a, i) = a[i]
write_cartesian!(a, v, i, j) = (a[i, j] = v; a)
write_linear!(a, v, i) = (a[i] = v; a)
component(a, k) = fieldview(a, k)

@testset "construction" begin
    data = reshape(collect(1.0:12.0), 4, 3)
    a = @inferred ViewStructArray{Point3D}(data)
    @test a isa ViewStructArray{Point3D, 1, Matrix{Float64}}
    @test size(a) == (4,)
    @test axes(a) == (Base.OneTo(4),)
    @test length(a) == 4
    @test eltype(a) == Point3D
    @test parent(a) === data
    @test IndexStyle(a) == IndexLinear()

    b = @inferred ViewStructArray{Point2D}(reshape(collect(Float32, 1:8), 4, 2))
    @test eltype(b) == Point2D{Float32}
    @test @inferred(ViewStructArray{Point2D{Float64}}(rand(3, 2))) isa ViewStructArray{Point2D{Float64}, 1}
    @test @inferred(ViewStructArray{Point2D, 2}(rand(3, 4, 2))) isa ViewStructArray{Point2D{Float64}, 2}

    c = ViewStructArray{RGBA}(reshape(collect(Float32, 1:16), 4, 4))
    @test c[1] == RGBA(1, 5, 9, 13)
    @test c[4] == RGBA(4, 8, 12, 16)

    z = ViewStructArray{Point3D}([1.0, 2.0, 3.0])
    @test size(z) == ()
    @test z[] == Point3D(1.0, 2.0, 3.0)

    big = ViewStructArray{Point2D}(BigFloat[1 3; 2 4])
    @test eltype(big) == Point2D{BigFloat}
    @test big[2].x == 2 && big[2].y == 4

    @test_throws "the components of Point3D have type Float64, but the parent has element type Int64" ViewStructArray{Point3D}(reshape(collect(1:12), 4, 3))
    @test_throws "the trailing size of the parent must be (3,) to hold the components of Point3D; got (2,)" ViewStructArray{Point3D}(rand(4, 2))
    @test_throws "needs at least 1 dims" ViewStructArray{Point3D}(fill(1.0))
    @test_throws "a ViewStructArray{Point3D, 1} needs a parent with 2 dims" ViewStructArray{Point3D, 1}(rand(2, 3, 3))
    @test_throws "the fields of Mixed must all have the same type" ViewStructArray{Mixed}(rand(3, 2))
    @test_throws "Float64 has no fields" ViewStructArray{Float64}(rand(3, 2))
    @test_throws "cannot build an element type like Point3D with components of type Float32" withcomponenttype(Point3D, Float32)
    @test_throws "element type Point2D must be concrete" ViewStructArray{Point2D, 1, Matrix{Float64}}(rand(3, 2))
end

@testset "element types" begin
    @test ncomponents(Point3D) == 3
    @test fieldshape(Point3D) == (3,)
    @test componenttype(Point2D{Float32}) == Float32
    @test componentnames(Point3D) == (:x, :y, :z)
    @test fromcomponents(Point3D, (1.0, 2.0, 3.0)) == Point3D(1.0, 2.0, 3.0)
    @test components(Point3D(1.0, 2.0, 3.0)) === (1.0, 2.0, 3.0)
    @test withcomponenttype(Point2D, Float64) == Point2D{Float64}
    @test withcomponenttype(Point2D{Float64}, Float32) == Point2D{Float32}
    @test withcomponenttype(Point3D, Float64) == Point3D
    @test isviewelement(Point3D)
    @test !isviewelement(Float64)
    @test !isviewelement(ComplexF64)
    @test !isviewelement(Mixed)
    @test !isviewelement(Point2D)
    @test !isviewelement(NTuple{2, Float64})
    @test !isviewelement(Base.RefValue{Float64})

    t = ViewStructArray{NTuple{2, Float64}}([1.0 3.0; 2.0 4.0])
    @test t[2] === (2.0, 4.0)
    nt = ViewStructArray{@NamedTuple{a::Float64, b::Float64}}([1.0 3.0; 2.0 4.0])
    @test nt[1] === (a = 1.0, b = 3.0)
    @test nt.b == [3.0, 4.0]
end

@testset "indexing" begin
    data = reshape(collect(1.0:24.0), 3, 4, 2)
    a = ViewStructArray{Point2D{Float64}}(data)
    @test size(a) == (3, 4)
    @test a[1, 1] == Point2D(1.0, 13.0)
    @test a[2, 1] == Point2D(2.0, 14.0)
    @test a[1, 2] == Point2D(4.0, 16.0)
    @test a[3, 4] == Point2D(12.0, 24.0)
    @test a[CartesianIndex(3, 4)] == a[3, 4]
    @test a[5] == a[2, 2]
    @test [a[i] for i in eachindex(IndexLinear(), a)] == vec([a[I] for I in CartesianIndices(a)])

    a[2, 3] = Point2D(100.0, 200.0)
    @test a[2, 3] == Point2D(100.0, 200.0)
    @test data[2, 3, 1] == 100.0 && data[2, 3, 2] == 200.0
    a[4] = Point2D(-1.0, -2.0)
    @test a[1, 2] == Point2D(-1.0, -2.0)

    v = ViewStructArray{Point3D}(reshape(collect(1.0:12.0), 4, 3))
    @test v[1] == Point3D(1.0, 5.0, 9.0)
    @test v[CartesianIndex(1)] == v[1]
    v[1] = Point3D(100.0, 200.0, 300.0)
    @test parent(v)[1, :] == [100.0, 200.0, 300.0]

    @test_throws BoundsError v[0]
    @test_throws BoundsError v[5]
    @test_throws BoundsError a[4, 1]
    @test_throws BoundsError a[13]
    @test_throws BoundsError (a[0, 1] = Point2D(0.0, 0.0))

    @test collect(v) isa Vector{Point3D}
    @test count(_ -> true, v) == 4
    @test map(p -> p.x, v) == parent(v)[:, 1]
end

@testset "views and slices" begin
    data = reshape(collect(1.0:24.0), 3, 4, 2)
    a = ViewStructArray{Point2D{Float64}}(data)

    s = @inferred view(a, 2:3, :)
    @test s isa ViewStructArray{Point2D{Float64}, 2}
    @test parent(s) isa SubArray
    @test s == a[2:3, :]
    s[1, 1] = Point2D(-5.0, -6.0)
    @test a[2, 1] == Point2D(-5.0, -6.0)

    r = @inferred view(a, 2, 1:3)
    @test r isa ViewStructArray{Point2D{Float64}, 1}
    @test r == [a[2, j] for j in 1:3]

    c = @inferred getindex(a, 1:2, 3)
    @test c isa ViewStructArray{Point2D{Float64}, 1, Matrix{Float64}}
    @test c == [a[1, 3], a[2, 3]]
    c[1] = Point2D(0.0, 0.0)
    @test a[1, 3] != Point2D(0.0, 0.0)

    @test a[[true, false, true], 2] == [a[1, 2], a[3, 2]]
    @test a[2, :] == [a[2, j] for j in 1:4]
    @test a[1:5] == [a[i] for i in 1:5]

    @test_throws BoundsError view(a, 1:4, 1)
    @test_throws BoundsError a[1:4, 1]
end

@testset "views and reshapes as parents" begin
    big = reshape(collect(1.0:60.0), 5, 4, 3)
    pv = view(big, 2:4, :, 1:2)
    a = ViewStructArray{Point2D}(pv)
    @test size(a) == (3, 4)
    @test a[2, 3] == Point2D(big[3, 3, 1], big[3, 3, 2])
    @test a[5] == a[2, 2]
    a[2, 3] = Point2D(0.0, 1.0)
    @test big[3, 3, 1] == 0.0 && big[3, 3, 2] == 1.0
    @test IndexStyle(a) == IndexCartesian()

    stepped = ViewStructArray{Point3D}(view(reshape(collect(1.0:30.0), 10, 3), 1:2:9, :))
    @test stepped[3] == Point3D(5.0, 15.0, 25.0)

    rr = ViewStructArray{Point2D}(reshape(1.0:12.0, 3, 2, 2))
    @test rr[2, 2] == Point2D(5.0, 11.0)
    @test rr[5] == rr[2, 2]

    rv = ViewStructArray{Point3D}(reshape(view(collect(1.0:24.0), 1:12), 4, 3))
    @test rv[4] == Point3D(4.0, 8.0, 12.0)
    @test copy(rv) == rv
    @test parent(copy(rv)) isa Matrix{Float64}
end

@testset "fieldview and properties" begin
    data = reshape(collect(1.0:12.0), 4, 3)
    a = ViewStructArray{Point3D}(data)
    @test @inferred(fieldview(a, 1)) == [1.0, 2.0, 3.0, 4.0]
    @test fieldview(a, 2) == [5.0, 6.0, 7.0, 8.0]
    @test fieldview(a, :z) == [9.0, 10.0, 11.0, 12.0]
    @test @inferred(getproperty(a, :y)) == [5.0, 6.0, 7.0, 8.0]
    @test a.x isa SubArray
    @test propertynames(a) == (:x, :y, :z)
    fieldview(a, 1)[1] = 100.0
    @test a[1].x == 100.0
    a.z .= 0
    @test all(p -> p.z == 0, a)

    @test_throws "component index 0 is out of range 1:3 for Point3D" fieldview(a, 0)
    @test_throws "component index 4 is out of range 1:3 for Point3D" fieldview(a, 4)
    @test_throws "Point3D has no component named :w; component names are (:x, :y, :z)" a.w
    @test_throws "component indices (1, 1) do not index the trailing size (3,)" fieldview(a, 1, 1)
end

@testset "similar, copy, show" begin
    data = reshape(collect(1.0:12.0), 4, 3)
    a = ViewStructArray{Point3D}(data)

    s1 = @inferred similar(a)
    @test s1 isa ViewStructArray{Point3D, 1, Matrix{Float64}}
    @test size(s1) == (4,)
    s2 = @inferred similar(a, Point3D, (2, 3))
    @test s2 isa ViewStructArray{Point3D, 2, Array{Float64, 3}}
    @test size(parent(s2)) == (2, 3, 3)
    s3 = @inferred similar(a, Point2D{Float32})
    @test s3 isa ViewStructArray{Point2D{Float32}, 1, Matrix{Float32}}
    s4 = @inferred similar(a, Float64)
    @test s4 isa Vector{Float64}
    s5 = @inferred similar(a, Mixed, (2,))
    @test s5 isa Vector{Mixed}

    c = @inferred copy(a)
    @test c == a
    @test parent(c) !== data

    @test summary(a) == "4-element ViewStructArray{Point3D}(::Matrix{Float64}) with eltype Point3D"
    @test occursin("Point3D(1.0, 5.0, 9.0)", sprint(show, MIME"text/plain"(), a))
    @test Base.mightalias(a, data)
    @test !Base.mightalias(a, c)
end

@testset "elementview" begin
    P = rand(3, 4, 2)
    @test @inferred(elementview(Point2D, P)) isa ViewStructArray{Point2D{Float64}, 2}
    @test @inferred(elementview(Float64, P)) === P
    @test elementview(Real, P) === P
    a = ViewStructArray{Point2D}(P)
    @test @inferred(elementview(a)) === a
    @test @inferred(elementview(P)) === P
end

@testset "inference, allocation and JET for element access" begin
    a = ViewStructArray{Point2D}(rand(3, 4, 2))
    v = Point2D(1.0, 2.0)
    read_cartesian(a, 2, 3); read_linear(a, 5); write_cartesian!(a, v, 2, 3); write_linear!(a, v, 5)
    @test @inferred(read_cartesian(a, 2, 3)) isa Point2D{Float64}
    @test @inferred(read_linear(a, 5)) isa Point2D{Float64}
    @test @inferred(write_cartesian!(a, v, 2, 3)) === a
    @test @inferred(write_linear!(a, v, 5)) === a
    @test (@allocated read_cartesian(a, 2, 3)) == 0
    @test (@allocated read_linear(a, 5)) == 0
    @test (@allocated write_cartesian!(a, v, 2, 3)) == 0
    @test (@allocated write_linear!(a, v, 5)) == 0
    component(a, 2)
    @test (@allocated component(a, 2)) == 0
    JET.@test_opt target_modules = (ViewStructArrays,) read_cartesian(a, 2, 3)
    JET.@test_opt target_modules = (ViewStructArrays,) read_linear(a, 5)
    JET.@test_opt target_modules = (ViewStructArrays,) write_cartesian!(a, v, 2, 3)
    JET.@test_opt target_modules = (ViewStructArrays,) write_linear!(a, v, 5)
    JET.@test_opt target_modules = (ViewStructArrays,) ViewStructArray{Point2D}(rand(3, 4, 2))
end

@testset "complex components" begin
    P = ComplexF64[1 + 1im 3; 2 4 - 2im]
    a = ViewStructArray{Point2D}(P)
    @test eltype(a) == Point2D{ComplexF64}
    @test a[2] == Point2D(2.0 + 0im, 4.0 - 2im)
    a[1] = Point2D(0.0im, 1.0im)
    @test P[1, :] == [0.0im, 1.0im]
    c = ViewStructArray{ComplexF64}([1.0 3.0; 2.0 4.0])
    @test c == [1 + 3im, 2 + 4im]
end

@testset "Adapt" begin
    a = ViewStructArray{Point2D}(rand(3, 2))
    b = adapt(Array{Float32}, a)
    @test b isa ViewStructArray{Point2D{Float32}, 1, Matrix{Float32}}
    @test parent(b) ≈ parent(a)
    @test Adapt.parent_type(typeof(a)) == Matrix{Float64}
end
