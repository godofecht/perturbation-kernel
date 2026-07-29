# Conformance test for the Julia binding.
#
# The binding does no arithmetic, so what matters is that values cross
# the ABI boundary unchanged. Expected numbers are the ones the Rust,
# Python, C++, Zig and TypeScript suites assert on.

using Test
using PerturbationKernel

@testset "PerturbationKernel" begin
    @info "perturbation-kernel $(version()) (schema $(schema_version())), simd $(simd_path()), gpu $(gpu_available())"

    @testset "reference value" begin
        r = run(Markov(k = 5, theta_max = 0.3), Config(n = 262144, seed = 20260610))
        @test r.value === 0.8802871704101562
        @test occursin("tail_survival", r.json)
    end

    @testset "host backends agree bit for bit" begin
        a = run(Markov(k = 5, theta_max = 0.3),
                Config(n = 262144, seed = 20260610, backend = "scalar"))
        b = run(Markov(k = 5, theta_max = 0.3),
                Config(n = 262144, seed = 20260610, backend = "simd"))
        @test reinterpret(UInt64, a.value) == reinterpret(UInt64, b.value)
    end

    @testset "null intensity recovers the base state" begin
        c = Config(n = 10000, seed = 5)
        @test run(Markov(k = 5, theta_max = 0.0, start = 2, base_label = 2), c).value == 1.0
        @test run(Gaussian(base = [1.5, -2.0], sigma_max = 0.0), c).value == 0.0
    end

    @testset "estimator ranges" begin
        c = Config(n = 20000, seed = 3)
        pol = run(Bistable(x0 = 0.0, dt = 0.01, theta_max = 0.5), c).value
        @test -1.0 <= pol <= 1.0
        @test run(Gaussian(base = [0.0, 0.0], sigma_max = 0.3), c).value <= 0.0
        surv = run(Markov(k = 4, theta_max = 0.25), c).value
        @test 0.0 <= surv <= 1.0
    end

    @testset "errors are errors, not wrong numbers" begin
        @test_throws PerturbationKernel.KernelError run(
            Markov(k = 5, theta_max = 0.3), Config(n = 0, seed = 1))
        @test_throws PerturbationKernel.KernelError run(
            Markov(k = 5, theta_max = 0.3),
            Config(n = 1000, seed = 1, invariance_lambda = 1.0,
                   epsilon = 0.05, eta = 0.05,
                   observation_diameter = 1.0, obs_dim = 1))
        @test_throws PerturbationKernel.KernelError run(
            Markov(k = 0, theta_max = 0.3), Config(n = 16, seed = 1))
    end

    if gpu_available()
        @testset "gpu is bit-identical to the host" begin
            h = run(Markov(k = 5, theta_max = 0.3),
                    Config(n = 262144, seed = 20260610, backend = "scalar"))
            d = run(Markov(k = 5, theta_max = 0.3),
                    Config(n = 262144, seed = 20260610, backend = "gpu"))
            @test reinterpret(UInt64, h.value) == reinterpret(UInt64, d.value)
        end
    else
        @info "no compute device; skipping the gpu check"
    end
end
