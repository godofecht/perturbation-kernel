// perturbation-kernel: C++20 wrapper over the C ABI.
//
// Header-only. RAII on the report handle, exceptions on error, and the
// three built-in families as value types that serialise themselves.
// Nothing here computes: every number comes back from the Rust engine
// unchanged, so this binding inherits the same bit-identity guarantees.
//
//   #include <perturbation_kernel.hpp>
//
//   namespace pk = perturbation_kernel;
//   auto report = pk::Markov{.k = 5, .theta_max = 0.3}
//                     .run(pk::Config{.n = 262144, .seed = 20260610});
//   std::cout << report.value() << '\n';

#ifndef PERTURBATION_KERNEL_HPP
#define PERTURBATION_KERNEL_HPP

#include "perturbation_kernel.h"

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace perturbation_kernel {

// Thrown for every non-zero pk_err. `code` carries the raw value so a
// caller can distinguish a bad config from an unmet accuracy claim.
class Error : public std::runtime_error {
public:
  Error(pk_err code, const std::string &what)
      : std::runtime_error(what), code_(code) {}
  pk_err code() const noexcept { return code_; }

private:
  pk_err code_;
};

// Which evaluation path to use. The first four agree bit for bit; only
// `gpu_f32` is a different number, and it says so in the report.
enum class Backend { Auto, Scalar, Simd, Gpu, GpuF32 };

inline const char *to_string(Backend b) {
  switch (b) {
  case Backend::Scalar:
    return "scalar";
  case Backend::Simd:
    return "simd";
  case Backend::Gpu:
    return "gpu";
  case Backend::GpuF32:
    return "gpu_f32";
  case Backend::Auto:
  default:
    return "auto";
  }
}

// Run configuration (SCHEMA section 5).
//
// The four accuracy fields are all-or-nothing: supplying some but not
// all is rejected by the engine, because a partial claim would silently
// disable the sample-complexity floor rather than enforce a weaker one.
struct Config {
  std::uint64_t n = 1024;
  std::uint64_t seed = 0;
  Backend backend = Backend::Auto;

  std::optional<double> forward_l{};
  std::optional<double> invariance_lambda{};

  std::optional<double> epsilon{};
  std::optional<double> eta{};
  std::optional<double> observation_diameter{};
  std::optional<std::uint32_t> obs_dim{};

  std::string to_json() const {
    std::ostringstream o;
    o.precision(17);
    o << R"({"schema_version":"1.0.0","n":)" << n << R"(,"seed":)" << seed
      << R"(,"intensity":{"kind":"uniform_interval","params":{},)"
         R"("null_parameter":0.0})"
      << R"(,"reduction":{"order":"tree","leaf_order":"index"})"
      << R"(,"lipschitz":{)";
    bool first = true;
    if (forward_l) {
      o << R"("forward_l":)" << *forward_l;
      first = false;
    }
    if (invariance_lambda) {
      if (!first)
        o << ',';
      o << R"("invariance_lambda":)" << *invariance_lambda;
    }
    o << '}';
    if (epsilon && eta && observation_diameter && obs_dim) {
      o << R"(,"accuracy":{"epsilon":)" << *epsilon << R"(,"eta":)" << *eta
        << R"(,"observation_diameter":)" << *observation_diameter
        << R"(,"obs_dim":)" << *obs_dim << '}';
    }
    if (backend != Backend::Auto)
      o << R"(,"backend":")" << to_string(backend) << '"';
    o << '}';
    return o.str();
  }
};

// Result of a run (SCHEMA section 6). Owns the underlying handle.
class Report {
public:
  explicit Report(pk_report *raw) : handle_(raw, &pk_free_report) {}

  double value() const { return pk_report_value(handle_.get()); }
  std::string json() const { return pk_report_json(handle_.get()); }

  explicit operator double() const { return value(); }

private:
  std::unique_ptr<pk_report, void (*)(pk_report *)> handle_;
};

namespace detail {

inline Report run_family(const std::string &family_json, const Config &cfg) {
  int err = 0;
  pk_report *raw =
      pk_run_family(family_json.c_str(), cfg.to_json().c_str(), &err);
  if (raw == nullptr) {
    static const char *names[] = {"ok",
                                  "invalid config",
                                  "null parameter mismatch",
                                  "sample-complexity floor not met",
                                  "empty ensemble",
                                  "panic caught at the ABI boundary"};
    const char *msg = (err >= 0 && err <= 5) ? names[err] : "unknown error";
    throw Error(static_cast<pk_err>(err),
                std::string("perturbation-kernel: ") + msg);
  }
  return Report(raw);
}

} // namespace detail

// Gaussian shift in R^d. Invariance is the negative empirical
// dispersion, so a larger value means a more stable result.
struct Gaussian {
  std::vector<double> base;
  double sigma_max = 0.0;

  std::string to_json() const {
    std::ostringstream o;
    o.precision(17);
    o << R"({"family":"gaussian","base":[)";
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (i)
        o << ',';
      o << base[i];
    }
    o << R"(],"sigma_max":)" << sigma_max << '}';
    return o.str();
  }
  Report run(const Config &cfg) const {
    return detail::run_family(to_json(), cfg);
  }
};

// Bistable double-well marble. Invariance is the polarisation, in
// [-1, 1]. Start at x0 = 0 to sit on the ridge, where the perturbation
// actually decides the outcome.
struct Bistable {
  double x0 = 0.0;
  double dt = 0.01;
  double theta_max = 0.0;

  std::string to_json() const {
    std::ostringstream o;
    o.precision(17);
    o << R"({"family":"bistable","x0":)" << x0 << R"(,"dt":)" << dt
      << R"(,"theta_max":)" << theta_max << '}';
    return o.str();
  }
  Report run(const Config &cfg) const {
    return detail::run_family(to_json(), cfg);
  }
};

// Finite-state chain under epsilon-uniform mixing. Invariance is the
// survival probability of base_label, in [0, 1]. This is the family the
// exact GPU backend supports.
struct Markov {
  std::uint32_t k = 2;
  double theta_max = 0.0;
  std::uint32_t start = 0;
  std::uint32_t base_label = 0;

  std::string to_json() const {
    std::ostringstream o;
    o.precision(17);
    o << R"({"family":"markov","k":)" << k << R"(,"start":)" << start
      << R"(,"base_label":)" << base_label << R"(,"theta_max":)" << theta_max
      << '}';
    return o.str();
  }
  Report run(const Config &cfg) const {
    return detail::run_family(to_json(), cfg);
  }
};

inline std::string version() { return pk_version(); }
inline std::string schema_version() { return pk_schema_version(); }
inline std::string simd_path() { return pk_simd_path(); }
inline bool gpu_available() { return pk_gpu_available() != 0; }

} // namespace perturbation_kernel

#endif // PERTURBATION_KERNEL_HPP
