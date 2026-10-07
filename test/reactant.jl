using Reactant
Reactant.set_default_backend("cpu")
using Enzyme

stokes_map(s) = Stokes(s.I * s.Q, s.Q + s.U, s.U * s.V, exp(s.I))
stokes_all(a) = stokes_map.(a)
stokes_wrap(P) = stokes_map.(FieldDimArray{Stokes}(P))
stokes_into!(d, a) = (d .= stokes_map.(a); d)
stokes_scalar(a) = (s -> s.I * s.V).(a)
stokes_weighted(a, w) = ((s, x) -> Stokes(s.I * x, s.Q, s.U, s.V * x)).(a, w)
point_all(a) = rotate.(a)
svector_all(a) = (v -> SVector(v[1] * v[2], v[2] - v[3], exp(v[3]))).(a)
stokes_loss(P) = sum(abs2, parent(stokes_map.(FieldDimArray{Stokes}(P))))
stokes_gradient(P) = Enzyme.gradient(Enzyme.Reverse, stokes_loss, P)[1]

function stokes_loss_gradient(P)
    I, Q, U, V = (P[:, :, k] for k in 1:4)
    return cat(
        2 .* I .* Q .^ 2 .+ 2 .* exp.(2 .* I),
        2 .* I .^ 2 .* Q .+ 2 .* (Q .+ U),
        2 .* (Q .+ U) .+ 2 .* U .* V .^ 2,
        2 .* U .^ 2 .* V;
        dims = 3
    )
end

shifted_fft(a) = fftshift(fft(a))
phase_scaled(a, x) = @. a * cospi(x * x')
ifft_into!(a) = (ifft!(a); a)

hlo_text(f, args...) = repr(Reactant.@code_hlo f(args...))
has_while(text) = occursin("stablehlo.while", text)
gathers(text) = count("all-gather(", text)
result_slices(a) = parent(a).sharding.device_to_array_slices

@testset "Reactant" begin
    @test Reactant.XLA.REACTANT_XLA_RUNTIME == "IFRT"
    ndev = length(Reactant.devices())
    @test ndev == 4

    @testset "tracing" begin
        Ph = rand(8, 6, 4)
        ah = FieldDimArray{Stokes}(Ph)
        ar = Reactant.to_rarray(ah)
        @test ar isa FieldDimArray{Stokes{Float64}, 2, <:Reactant.ConcreteIFRTArray{Float64, 3}}
        @test Array(parent(ar)) == Ph
        T = Reactant.traced_type(typeof(ar), Val(Reactant.ConcreteToTraced), Union{}, Reactant.Sharding.NoSharding(), nothing)
        @test T == FieldDimArray{Stokes{Reactant.TracedRNumber{Float64}}, 2, Reactant.TracedRArray{Float64, 3}}
        r = @jit identity(ar)
        @test r isa FieldDimArray{Stokes{Float64}, 2}
        @test Array(parent(r)) == Ph
    end

    @testset "broadcasts equal the host" begin
        Ph = rand(8, 6, 4)
        ah = FieldDimArray{Stokes}(Ph)
        ar = Reactant.to_rarray(ah)
        Pr = Reactant.to_rarray(Ph)
        wh = rand(8, 6)
        wr = Reactant.to_rarray(wh)
        ref = parent(stokes_all(ah))

        r = @jit stokes_all(ar)
        @test r isa FieldDimArray{Stokes{Float64}, 2}
        @test Array(parent(r)) ≈ ref
        @test Array(parent(@jit stokes_wrap(Pr))) ≈ ref
        @test Array(@jit stokes_scalar(ar)) ≈ stokes_scalar(ah)
        @test Array(parent(@jit stokes_weighted(ar, wr))) ≈ parent(stokes_weighted(ah, wh))
        d = Reactant.to_rarray(similar(ah))
        @test Array(parent(@jit stokes_into!(d, ar))) ≈ ref

        ph = FieldDimArray{Point2D}(rand(5, 2))
        @test Array(parent(@jit point_all(Reactant.to_rarray(ph)))) ≈ parent(point_all(ph))

        vh = FieldDimArray{SVector{3}}(rand(7, 3))
        @test Array(parent(@jit svector_all(Reactant.to_rarray(vh)))) ≈ parent(svector_all(vh))

        A, X, B = (FieldDimArray{SMatrix{2, 2}}(rand(8, 2, 2)) for _ in 1:3)
        Ar, Xr, Br = Reactant.to_rarray.((A, X, B))
        m = @jit sandwich(Ar, Xr, Br)
        @test m isa FieldDimArray{SMatrix{2, 2, Float64, 4}, 1}
        @test Array(parent(m)) ≈ parent(sandwich(A, X, B))
        D = Reactant.to_rarray(similar(A))
        @test Array(parent(@jit sandwich_into!(D, Ar, Xr, Br))) ≈ parent(sandwich(A, X, B))

        @test !has_while(hlo_text(stokes_all, ar))
        @test !has_while(hlo_text(stokes_scalar, ar))
        @test !has_while(hlo_text(stokes_into!, d, ar))
        @test !has_while(hlo_text(sandwich, Ar, Xr, Br))
        @test !has_while(hlo_text(sandwich_into!, D, Ar, Xr, Br))
    end

    @testset "subtrees without a FieldDimArray are evaluated once" begin
        x = range(-1.0, 1.0; length = 8)
        a = FieldDimArray{Stokes}(rand(8, 8, 4))
        ar = Reactant.to_rarray(a)
        @test Array(parent(@jit(phase_scaled(ar, x)))) ≈ parent(phase_scaled(a, x))
        @test !occursin("stablehlo.cosine", hlo_text(phase_scaled, ar, x))
    end

    @testset "FFTs" begin
        P = rand(8, 6, 2)
        a = FieldDimArray{Point2D}(P)
        ar = Reactant.to_rarray(a)
        @test parent(@jit(shifted_fft(ar))) ≈ parent(shifted_fft(a))
        text = hlo_text(shifted_fft, ar)
        @test count("stablehlo.fft", text) == 1
        @test !has_while(text)
        br = Reactant.to_rarray(fft(a))
        @test parent(@jit(ifft_into!(br))) ≈ P
    end

    @testset "sharding" begin
        mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices(), ndev), (:d,))
        shard(P, dim) = Reactant.to_rarray(P; sharding = Reactant.Sharding.DimsSharding(mesh, (dim,), (:d,)))
        split(n) = [(1:(n ÷ ndev)) .+ k * (n ÷ ndev) for k in 0:(ndev - 1)]

        @testset "Stokes elements along a leading dim" begin
            Ph = rand(8, 6, 4)
            as = FieldDimArray{Stokes}(shard(Ph, 1))
            r = @jit stokes_all(as)
            @test Array(parent(r)) ≈ parent(stokes_all(FieldDimArray{Stokes}(Ph)))
            @test sort(unique(first.(result_slices(r)))) == split(8)
            @test all(s -> s[2] == 1:6 && s[3] == 1:4, result_slices(r))
            @test gathers(repr(@code_xla shardy_passes = :to_mhlo_shardings stokes_all(as))) == 0

        end

        @testset "Stokes elements along a non-leading dim" begin
            Ph = rand(6, 8, 4)
            Ps = shard(Ph, 2)
            r2 = @jit stokes_wrap(Ps)
            @test Array(parent(r2)) ≈ parent(stokes_all(FieldDimArray{Stokes}(Ph)))
            @test sort(unique(map(s -> s[2], result_slices(r2)))) == split(8)
            @test all(s -> s[1] == 1:6 && s[3] == 1:4, result_slices(r2))
            @test gathers(repr(@code_xla shardy_passes = :to_mhlo_shardings stokes_wrap(Ps))) == 0
        end

        @testset "2×2 elements along a leading dim" begin
            A, X, B = (FieldDimArray{SMatrix{2, 2}}(rand(8, 2, 2)) for _ in 1:3)
            As, Xs, Bs = (FieldDimArray{SMatrix{2, 2}}(shard(parent(M), 1)) for M in (A, X, B))
            m = @jit sandwich(As, Xs, Bs)
            @test Array(parent(m)) ≈ parent(sandwich(A, X, B))
            @test sort(unique(first.(result_slices(m)))) == split(8)
            @test all(s -> s[2] == 1:2 && s[3] == 1:2, result_slices(m))
            xla = repr(@code_xla shardy_passes = :to_mhlo_shardings sandwich(As, Xs, Bs))
            @test gathers(xla) == 0
            @test !occursin("scatter", xla)

            Ds = FieldDimArray{SMatrix{2, 2}}(shard(zeros(8, 2, 2), 1))
            mi = @jit sandwich_into!(Ds, As, Xs, Bs)
            @test Array(parent(mi)) ≈ parent(sandwich(A, X, B))
            @test sort(unique(first.(result_slices(mi)))) == split(8)
            @test gathers(repr(@code_xla shardy_passes = :to_mhlo_shardings sandwich_into!(Ds, As, Xs, Bs))) == 0
        end

        @testset "Enzyme reverse gradient" begin
            Ph = rand(8, 8, 4)
            expected = stokes_loss_gradient(Ph)
            @test Array(@jit stokes_gradient(Reactant.to_rarray(Ph))) ≈ expected
            g1 = @jit stokes_gradient(shard(Ph, 1))
            @test Array(g1) ≈ expected
            g2 = @jit stokes_gradient(shard(Ph, 2))
            @test all(isfinite, Array(g2))
            @test Array(g2) ≈ expected
        end
    end
end
