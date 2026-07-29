// Conformance test for the C++ binding.
//
// The binding does no arithmetic, so what matters is that values cross
// the ABI boundary unchanged. Every expected number here is the one the
// Rust and Python suites assert on, so a mismatch means the boundary
// lost something rather than the estimator being wrong.

#include <perturbation_kernel.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace pk = perturbation_kernel;

static int failures = 0;

static void check(bool ok, const std::string &what) {
  std::printf("  %-58s %s\n", what.c_str(), ok ? "ok" : "FAILED");
  if (!ok)
    ++failures;
}

int main() {
  std::printf("perturbation-kernel %s (schema %s), simd path %s, gpu %s\n\n",
              pk::version().c_str(), pk::schema_version().c_str(),
              pk::simd_path().c_str(), pk::gpu_available() ? "yes" : "no");

  // The canonical value, identical to the Rust and Python suites.
  const double expected = 0.8802871704101562;
  {
    auto r = pk::Markov{.k = 5, .theta_max = 0.3}.run(
        pk::Config{.n = 262144, .seed = 20260610});
    check(r.value() == expected, "markov matches the reference value exactly");
    check(r.json().find("tail_survival") != std::string::npos,
          "report json carries the functional tag");
  }

  // The host backends must agree bit for bit across the boundary too.
  {
    auto a = pk::Markov{.k = 5, .theta_max = 0.3}.run(
        pk::Config{.n = 262144, .seed = 20260610, .backend = pk::Backend::Scalar});
    auto b = pk::Markov{.k = 5, .theta_max = 0.3}.run(
        pk::Config{.n = 262144, .seed = 20260610, .backend = pk::Backend::Simd});
    check(a.value() == b.value(), "scalar and simd agree bit for bit");
    check(a.value() == expected, "scalar matches the reference value");
  }

  // C2: at null intensity the perturbation is the identity.
  {
    auto r = pk::Markov{.k = 5, .theta_max = 0.0, .start = 2, .base_label = 2}
                 .run(pk::Config{.n = 10000, .seed = 5});
    check(r.value() == 1.0, "null intensity recovers the base label exactly");
    auto g = pk::Gaussian{.base = {1.5, -2.0}, .sigma_max = 0.0}.run(
        pk::Config{.n = 10000, .seed = 5});
    check(g.value() == 0.0, "null intensity gives zero dispersion");
  }

  // Ranges.
  {
    auto b = pk::Bistable{.x0 = 0.0, .dt = 0.01, .theta_max = 0.5}.run(
        pk::Config{.n = 20000, .seed = 3});
    check(b.value() >= -1.0 && b.value() <= 1.0, "polarisation lies in [-1, 1]");
    auto g = pk::Gaussian{.base = {0.0, 0.0}, .sigma_max = 0.3}.run(
        pk::Config{.n = 20000, .seed = 3});
    check(g.value() <= 0.0, "negative dispersion is non-positive");
  }

  // Errors must arrive as exceptions carrying the right code, not as
  // silently wrong numbers.
  {
    bool threw = false;
    try {
      pk::Markov{.k = 5, .theta_max = 0.3}.run(pk::Config{.n = 0, .seed = 1});
    } catch (const pk::Error &e) {
      threw = e.code() == PK_EMPTY_ENSEMBLE;
    }
    check(threw, "an empty ensemble throws PK_EMPTY_ENSEMBLE");
  }
  {
    bool threw = false;
    try {
      pk::Markov{.k = 5, .theta_max = 0.3}.run(pk::Config{
          .n = 1000,
          .seed = 1,
          .invariance_lambda = 1.0,
          .epsilon = 0.05,
          .eta = 0.05,
          .observation_diameter = 1.0,
          .obs_dim = 1,
      });
    } catch (const pk::Error &e) {
      threw = e.code() == PK_SAMPLE_FLOOR;
    }
    check(threw, "an unsupported accuracy claim throws PK_SAMPLE_FLOOR");
  }
  {
    bool threw = false;
    try {
      pk::Markov{.k = 0, .theta_max = 0.3}.run(pk::Config{.n = 16, .seed = 1});
    } catch (const pk::Error &) {
      threw = true;
    }
    check(threw, "an out-of-domain family throws");
  }

  // The exact device backend, when there is a device.
  if (pk::gpu_available()) {
    auto host = pk::Markov{.k = 5, .theta_max = 0.3}.run(
        pk::Config{.n = 262144, .seed = 20260610, .backend = pk::Backend::Scalar});
    auto dev = pk::Markov{.k = 5, .theta_max = 0.3}.run(
        pk::Config{.n = 262144, .seed = 20260610, .backend = pk::Backend::Gpu});
    check(host.value() == dev.value(), "gpu is bit-identical to the host");
  } else {
    std::printf("  %-58s skipped\n", "gpu (no device on this machine)");
  }

  std::printf("\n%s\n", failures ? "FAILURES" : "all checks passed");
  return failures ? 1 : 0;
}
