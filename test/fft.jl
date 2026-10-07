using FFTW

componentwise_fft(f, P, region) = cat((f(selectdim(P, ndims(P), k), region) for k in axes(P, ndims(P)))...; dims = ndims(P))

@testset "FFTs" begin
    P = rand(8, 6, 2)
    a = FieldDimArray{Point2D}(P)
    b = fft(a)
    @test b isa FieldDimArray{Point2D{ComplexF64}, 2}
    @test parent(b) ≈ componentwise_fft(fft, P, 1:2)
    @test parent(fft(a, 1)) ≈ componentwise_fft(fft, P, 1)
    @test parent(ifft(b)) ≈ P
    @test parent(bfft(b)) ≈ componentwise_fft(bfft, parent(b), 1:2)

    c = copy(b)
    @test parent(fft!(c)) === parent(c)
    @test parent(c) ≈ parent(fft(b))
    c = copy(b)
    ifft!(c)
    @test parent(c) ≈ parent(ifft(b))

    p = plan_fft(b)
    @test parent(p * b) ≈ parent(fft(b))
    @test parent(p \ (p * b)) ≈ parent(b)
    @test parent(inv(p) * (p * b)) ≈ parent(b)
    @test_throws "region 1:3 is not within the 2 dims of the FieldDimArray" fft(a, 1:3)

    s = fftshift(b)
    @test s isa FieldDimArray{Point2D{ComplexF64}, 2}
    @test parent(s) == fftshift(parent(b), 1:2)
    @test parent(fftshift(b, 1)) == fftshift(parent(b), 1)
    @test parent(ifftshift(s)) == parent(b)

    m = FieldDimArray{SMatrix{2, 2, Float64, 4}}(rand(8, 6, 2, 2))
    @test parent(fft(m)) ≈ fft(parent(m), 1:2)
    @test parent(fftshift(m)) == fftshift(parent(m), 1:2)
end
