rotate(p::Point2D) = Point2D(p.x * p.y, p.x + p.y)
rotate_all(a) = rotate.(a)
rotate_into!(d, a) = (d .= rotate.(a); d)
xy_all(a) = (p -> p.x * p.y).(a)
xy_into!(d, a) = (d .= (p -> p.x * p.y).(a); d)
mixed_all(a, w) = (p -> p.x * p.y).(a) .+ w .* 2
shift_all(a, w) = ((p, s) -> Point2D(p.x + s, p.y - s)).(a, w)
sandwich(A, X, B) = A .* X .* adjoint.(B)
sandwich_into!(D, A, X, B) = (D .= A .* X .* adjoint.(B); D)
rotate_stokes(s) = Stokes(s.I, s.Q * s.U, s.U - s.Q, s.V)

materialized(f, args...) = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
componentwise_copy(bc) = FieldDimArrays._componentwise_copy(bc)
componentwise_copyto!(dest, bc) = FieldDimArrays._componentwise_copyto!(dest, bc)

@testset "broadcasting" begin
    P = rand(3, 4, 2)
    a = FieldDimArray{Point2D}(P)
    w = rand(3, 4)
    ref = [rotate(a[I]) for I in CartesianIndices(a)]

    @testset "style" begin
        @test Broadcast.BroadcastStyle(typeof(a)) === FieldDimArrays.FieldDimArrayStyle{2}()
        @test Broadcast.combine_styles(a, w) === FieldDimArrays.FieldDimArrayStyle{2}()
        @test Broadcast.combine_styles(a, rand(3, 4, 5)) === FieldDimArrays.FieldDimArrayStyle{3}()
        @test Broadcast.combine_styles(a, 1.0) === FieldDimArrays.FieldDimArrayStyle{2}()
        @test isempty(Test.detect_ambiguities(FieldDimArrays; recursive = true))
    end

    @testset "out of place" begin
        r = @inferred rotate_all(a)
        @test r isa FieldDimArray{Point2D{Float64}, 2, Array{Float64, 3}}
        @test r == ref
        s = @inferred xy_all(a)
        @test s isa Matrix{Float64}
        @test s == P[:, :, 1] .* P[:, :, 2]
        m = @inferred mixed_all(a, w)
        @test m isa Matrix{Float64}
        @test m ≈ s .+ 2w
        sh = @inferred shift_all(a, w)
        @test sh isa FieldDimArray{Point2D{Float64}, 2}
        @test sh == [Point2D(a[I].x + w[I], a[I].y - w[I]) for I in CartesianIndices(a)]
        c = (p -> p.x + 1im * p.y).(a)
        @test c isa Matrix{ComplexF64}
        mx = (p -> Mixed(p.x, 1)).(a)
        @test mx isa Matrix{Mixed}
        @test (a .== a) == trues(3, 4)
        JET.@test_opt target_modules = (FieldDimArrays,) rotate_all(a)
        JET.@test_opt target_modules = (FieldDimArrays,) xy_all(a)
    end

    @testset "in place" begin
        d = similar(a)
        @test @inferred(rotate_into!(d, a)) === d
        @test d == ref
        ds = zeros(3, 4)
        @test @inferred(xy_into!(ds, a)) === ds
        @test ds == P[:, :, 1] .* P[:, :, 2]
        b = copy(a)
        b .= rotate.(b)
        @test b == ref
        b .= Ref(Point2D(1.0, 2.0))
        @test all(==(Point2D(1.0, 2.0)), b)
        b .= a
        @test b == a
        JET.@test_opt target_modules = (FieldDimArrays,) rotate_into!(d, a)
        JET.@test_opt target_modules = (FieldDimArrays,) xy_into!(ds, a)
    end

    @testset "static elements" begin
        A, X, B = (FieldDimArray{SMatrix{2, 2}}(rand(5, 2, 2)) for _ in 1:3)
        r = @inferred sandwich(A, X, B)
        @test r isa FieldDimArray{SMatrix{2, 2, Float64, 4}, 1, Array{Float64, 3}}
        @test all(i -> r[i] ≈ A[i] * X[i] * B[i]', eachindex(r))
        D = similar(A)
        @test @inferred(sandwich_into!(D, A, X, B)) == r
        JET.@test_opt target_modules = (FieldDimArrays,) sandwich(A, X, B)

        S = FieldDimArray{Stokes}(rand(6, 4))
        rs = rotate_stokes.(S)
        @test rs isa FieldDimArray{Stokes{Float64}, 1, Matrix{Float64}}
        @test all(i -> rs[i] == rotate_stokes(S[i]), eachindex(S))
        @test (s -> s.I).(S) == parent(S)[:, 1]
        v = (x -> SVector(x, 2x)).(rand(4))
        @test v isa Vector{SVector{2, Float64}}
    end

    @testset "component-wise lowering on the CPU" begin
        r = @inferred componentwise_copy(materialized(rotate, a))
        @test r isa FieldDimArray{Point2D{Float64}, 2, Array{Float64, 3}}
        @test r == ref
        s = @inferred componentwise_copy(materialized(p -> p.x * p.y, a))
        @test s == P[:, :, 1] .* P[:, :, 2]
        sh = componentwise_copy(materialized((p, s) -> Point2D(p.x + s, p.y - s), a, w))
        @test sh == shift_all(a, w)

        A, X, B = (FieldDimArray{SMatrix{2, 2}}(rand(5, 2, 2)) for _ in 1:3)
        m = @inferred componentwise_copy(materialized((a, x, b) -> a * x * b', A, X, B))
        @test m ≈ sandwich(A, X, B)
        D = FieldDimArray{SMatrix{2, 2}, 1}(zeros(5, 4))
        @test componentwise_copyto!(D, materialized((a, x, b) -> a * x * b', A, X, B)) ≈ sandwich(A, X, B)

        b = copy(a)
        componentwise_copyto!(b, materialized(rotate, b))
        @test b == ref
        ds = zeros(3, 4)
        componentwise_copyto!(ds, materialized(p -> p.x * p.y, a))
        @test ds == P[:, :, 1] .* P[:, :, 2]
        JET.@test_opt target_modules = (FieldDimArrays,) componentwise_copy(materialized(rotate, a))
        @test_throws DimensionMismatch componentwise_copyto!(similar(a, Point2D{Float64}, (2, 2)), materialized(rotate, a))
    end
end
