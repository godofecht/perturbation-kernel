"""
    PerturbationKernel

Julia binding over the perturbation-kernel C ABI.

Julia calls C directly through `ccall`, so this module is a thin layer:
it builds the JSON the ABI expects, turns error codes into exceptions,
and frees the report handle. Nothing here computes, so the binding
inherits the engine's bit-identity guarantees unchanged.

```julia
using PerturbationKernel

r = run(Markov(k = 5, theta_max = 0.3), Config(n = 262144, seed = 20260610))
r.value    # 0.8802871704101562
```

The shared library is located through `PK_LIBRARY`, falling back to
`libperturbation_kernel` on the default search path. Point it at the
artefact `cargo build --release` produced:

```julia
ENV["PK_LIBRARY"] = "/path/to/target/release/libperturbation_kernel.dylib"
```
"""
module PerturbationKernel

export Config, Gaussian, Bistable, Markov, Report
export version, schema_version, simd_path, gpu_available

const LIBPK = get(ENV, "PK_LIBRARY", "libperturbation_kernel")

"""Raised for every non-zero error code the ABI returns."""
struct KernelError <: Exception
    code::Int
    msg::String
end
Base.showerror(io::IO, e::KernelError) =
    print(io, "PerturbationKernel error ", e.code, ": ", e.msg)

const ERROR_NAMES = Dict(
    1 => "invalid config",
    2 => "null parameter mismatch",
    3 => "sample-complexity floor not met",
    4 => "empty ensemble",
    5 => "panic caught at the ABI boundary",
)

"""
    Config(; n, seed, backend, ...)

Run configuration (SCHEMA section 5).

The four accuracy fields are all-or-nothing: supplying some but not all
is rejected by the engine, because a partial claim would quietly disable
the sample-complexity floor rather than enforce a weaker one.
"""
Base.@kwdef struct Config
    n::Int = 1024
    seed::Int = 0
    backend::String = "auto"
    forward_l::Union{Float64,Nothing} = nothing
    invariance_lambda::Union{Float64,Nothing} = nothing
    epsilon::Union{Float64,Nothing} = nothing
    eta::Union{Float64,Nothing} = nothing
    observation_diameter::Union{Float64,Nothing} = nothing
    obs_dim::Union{Int,Nothing} = nothing
end

# A minimal JSON writer. Julia's stdlib has no JSON module, and a
# binding should not drag in a dependency to emit a handful of fields.
# `q` keeps the quoting readable: Julia's triple-quoted strings do not
# sit well next to literal double quotes.
const q = '"'
jsonnum(x::Float64) = repr(x)
jsonnum(x::Integer) = string(x)
field(name, value) = string(q, name, q, ":", value)

function to_json(c::Config)
    parts = [
        field("schema_version", string(q, "1.0.0", q)),
        field("n", c.n),
        field("seed", c.seed),
        field("intensity", string("{",
            field("kind", string(q, "uniform_interval", q)), ",",
            field("params", "{}"), ",",
            field("null_parameter", "0.0"), "}")),
        field("reduction", string("{",
            field("order", string(q, "tree", q)), ",",
            field("leaf_order", string(q, "index", q)), "}")),
    ]

    lip = String[]
    c.forward_l === nothing || push!(lip, field("forward_l", jsonnum(c.forward_l)))
    c.invariance_lambda === nothing ||
        push!(lip, field("invariance_lambda", jsonnum(c.invariance_lambda)))
    push!(parts, field("lipschitz", string("{", join(lip, ","), "}")))

    if c.epsilon !== nothing && c.eta !== nothing &&
       c.observation_diameter !== nothing && c.obs_dim !== nothing
        push!(parts, field("accuracy", string("{",
            field("epsilon", jsonnum(c.epsilon)), ",",
            field("eta", jsonnum(c.eta)), ",",
            field("observation_diameter", jsonnum(c.observation_diameter)), ",",
            field("obs_dim", c.obs_dim), "}")))
    end

    c.backend == "auto" || push!(parts, field("backend", string(q, c.backend, q)))
    string("{", join(parts, ","), "}")
end

"""Result of a run (SCHEMA section 6)."""
struct Report
    value::Float64
    json::String
end

abstract type Family end

"""Gaussian shift in R^d; the invariance is the negative empirical dispersion."""
Base.@kwdef struct Gaussian <: Family
    base::Vector{Float64}
    sigma_max::Float64 = 0.0
end
to_json(f::Gaussian) = string("{",
    field("family", string(q, "gaussian", q)), ",",
    field("base", string("[", join((jsonnum(x) for x in f.base), ","), "]")), ",",
    field("sigma_max", jsonnum(f.sigma_max)), "}")

"""Bistable double-well marble; the invariance is the polarisation in [-1, 1]."""
Base.@kwdef struct Bistable <: Family
    x0::Float64 = 0.0
    dt::Float64 = 0.01
    theta_max::Float64 = 0.0
end
to_json(f::Bistable) = string("{",
    field("family", string(q, "bistable", q)), ",",
    field("x0", jsonnum(f.x0)), ",",
    field("dt", jsonnum(f.dt)), ",",
    field("theta_max", jsonnum(f.theta_max)), "}")

"""Finite-state chain; the invariance is the survival probability in [0, 1]."""
Base.@kwdef struct Markov <: Family
    k::Int = 2
    theta_max::Float64 = 0.0
    start::Int = 0
    base_label::Int = 0
end
to_json(f::Markov) = string("{",
    field("family", string(q, "markov", q)), ",",
    field("k", f.k), ",",
    field("start", f.start), ",",
    field("base_label", f.base_label), ",",
    field("theta_max", jsonnum(f.theta_max)), "}")

"""
    run(family, config) -> Report

Evaluate `family` under `config`. Throws `KernelError` on any schema or
domain violation.
"""
function Base.run(f::Family, c::Config)
    err = Ref{Cint}(0)
    handle = ccall((:pk_run_family, LIBPK), Ptr{Cvoid},
                   (Cstring, Cstring, Ptr{Cint}),
                   to_json(f), to_json(c), err)
    if handle == C_NULL
        code = Int(err[])
        throw(KernelError(code, get(ERROR_NAMES, code, "unknown error")))
    end
    try
        value = ccall((:pk_report_value, LIBPK), Cdouble, (Ptr{Cvoid},), handle)
        json = unsafe_string(
            ccall((:pk_report_json, LIBPK), Cstring, (Ptr{Cvoid},), handle))
        Report(value, json)
    finally
        ccall((:pk_free_report, LIBPK), Cvoid, (Ptr{Cvoid},), handle)
    end
end

version() = unsafe_string(ccall((:pk_version, LIBPK), Cstring, ()))
schema_version() = unsafe_string(ccall((:pk_schema_version, LIBPK), Cstring, ()))

"""Host vector path: "scalar", "neon" or "avx2". Informational only."""
simd_path() = unsafe_string(ccall((:pk_simd_path, LIBPK), Cstring, ()))

gpu_available() = ccall((:pk_gpu_available, LIBPK), Cint, ()) != 0

end # module
