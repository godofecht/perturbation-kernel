//! perturbation-kernel: Zig binding over the C ABI.
//!
//! Zig imports the C header directly, so this module is a thin layer:
//! it builds the JSON the ABI expects, turns error codes into a Zig
//! error set, and frees the report handle. Nothing here computes, so the
//! binding inherits the engine's bit-identity guarantees unchanged.
//!
//!     const pk = @import("perturbation_kernel.zig");
//!
//!     var report = try pk.Markov{ .k = 5, .theta_max = 0.3 }
//!         .run(allocator, .{ .n = 262144, .seed = 20260610 });
//!     defer report.deinit();
//!     std.debug.print("{d}\n", .{report.value()});

const std = @import("std");

const c = @cImport({
    @cInclude("perturbation_kernel.h");
});

pub const Error = error{
    InvalidConfig,
    NullParameterMismatch,
    /// The asserted accuracy needs more draws than `n` provides
    /// (Theorem 7.3(c)).
    SampleFloor,
    EmptyEnsemble,
    /// A panic was caught at the ABI boundary.
    Panic,
    Unknown,
};

fn errorFrom(code: c_int) Error {
    return switch (code) {
        1 => Error.InvalidConfig,
        2 => Error.NullParameterMismatch,
        3 => Error.SampleFloor,
        4 => Error.EmptyEnsemble,
        5 => Error.Panic,
        else => Error.Unknown,
    };
}

/// Which evaluation path to use. The first four agree bit for bit; only
/// `gpu_f32` is a different number, and it says so in the report.
pub const Backend = enum {
    auto,
    scalar,
    simd,
    gpu,
    gpu_f32,

    pub fn text(self: Backend) []const u8 {
        return switch (self) {
            .auto => "auto",
            .scalar => "scalar",
            .simd => "simd",
            .gpu => "gpu",
            .gpu_f32 => "gpu_f32",
        };
    }
};

/// Run configuration (SCHEMA section 5).
///
/// The four accuracy fields are all-or-nothing. Supplying some but not
/// all is rejected by the engine, because a partial claim would quietly
/// disable the sample-complexity floor rather than enforce a weaker one.
pub const Config = struct {
    n: u64 = 1024,
    seed: u64 = 0,
    backend: Backend = .auto,

    forward_l: ?f64 = null,
    invariance_lambda: ?f64 = null,

    epsilon: ?f64 = null,
    eta: ?f64 = null,
    observation_diameter: ?f64 = null,
    obs_dim: ?u32 = null,

    pub fn toJson(self: Config, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayList(u8).init(allocator);
        errdefer buf.deinit();
        const w = buf.writer();

        try w.print(
            \\{{"schema_version":"1.0.0","n":{d},"seed":{d},
            \\"intensity":{{"kind":"uniform_interval","params":{{}},"null_parameter":0.0}},
            \\"reduction":{{"order":"tree","leaf_order":"index"}},"lipschitz":{{
        , .{ self.n, self.seed });

        var wrote = false;
        if (self.forward_l) |v| {
            try w.print("\"forward_l\":{d}", .{v});
            wrote = true;
        }
        if (self.invariance_lambda) |v| {
            if (wrote) try w.writeAll(",");
            try w.print("\"invariance_lambda\":{d}", .{v});
        }
        try w.writeAll("}");

        if (self.epsilon != null and self.eta != null and
            self.observation_diameter != null and self.obs_dim != null)
        {
            try w.print(
                \\,"accuracy":{{"epsilon":{d},"eta":{d},"observation_diameter":{d},"obs_dim":{d}}}
            , .{
                self.epsilon.?,
                self.eta.?,
                self.observation_diameter.?,
                self.obs_dim.?,
            });
        }
        if (self.backend != .auto) {
            try w.print(",\"backend\":\"{s}\"", .{self.backend.text()});
        }
        try w.writeAll("}");
        return buf.toOwnedSlice();
    }
};

/// Result of a run (SCHEMA section 6). Owns the underlying handle.
pub const Report = struct {
    handle: *c.pk_report,

    pub fn deinit(self: *Report) void {
        c.pk_free_report(self.handle);
    }

    /// The scalar estimate.
    pub fn value(self: Report) f64 {
        return c.pk_report_value(self.handle);
    }

    /// The full report as JSON. Borrowed from the handle; invalid after
    /// `deinit`.
    pub fn json(self: Report) []const u8 {
        return std.mem.span(c.pk_report_json(self.handle));
    }
};

fn runFamily(allocator: std.mem.Allocator, family_json: []const u8, cfg: Config) !Report {
    const cfg_json = try cfg.toJson(allocator);
    defer allocator.free(cfg_json);

    const fam_z = try allocator.dupeZ(u8, family_json);
    defer allocator.free(fam_z);
    const cfg_z = try allocator.dupeZ(u8, cfg_json);
    defer allocator.free(cfg_z);

    var err: c_int = 0;
    const handle = c.pk_run_family(fam_z.ptr, cfg_z.ptr, &err);
    if (handle == null) return errorFrom(err);
    return Report{ .handle = handle.? };
}

/// Gaussian shift in R^d. The invariance is the negative empirical
/// dispersion, so a larger value means a more stable result.
pub const Gaussian = struct {
    base: []const f64,
    sigma_max: f64 = 0.0,

    pub fn run(self: Gaussian, allocator: std.mem.Allocator, cfg: Config) !Report {
        var buf = std.ArrayList(u8).init(allocator);
        defer buf.deinit();
        const w = buf.writer();
        try w.writeAll("{\"family\":\"gaussian\",\"base\":[");
        for (self.base, 0..) |x, i| {
            if (i != 0) try w.writeAll(",");
            try w.print("{d}", .{x});
        }
        try w.print("],\"sigma_max\":{d}}}", .{self.sigma_max});
        return runFamily(allocator, buf.items, cfg);
    }
};

/// Bistable double-well marble. The invariance is the polarisation, in
/// [-1, 1]. Start at `x0 = 0` to sit on the ridge, where the
/// perturbation actually decides the outcome.
pub const Bistable = struct {
    x0: f64 = 0.0,
    dt: f64 = 0.01,
    theta_max: f64 = 0.0,

    pub fn run(self: Bistable, allocator: std.mem.Allocator, cfg: Config) !Report {
        const json = try std.fmt.allocPrint(
            allocator,
            "{{\"family\":\"bistable\",\"x0\":{d},\"dt\":{d},\"theta_max\":{d}}}",
            .{ self.x0, self.dt, self.theta_max },
        );
        defer allocator.free(json);
        return runFamily(allocator, json, cfg);
    }
};

/// Finite-state chain under epsilon-uniform mixing. The invariance is
/// the survival probability of `base_label`, in [0, 1]. This is the
/// family the exact GPU backend supports.
pub const Markov = struct {
    k: u32 = 2,
    theta_max: f64 = 0.0,
    start: u32 = 0,
    base_label: u32 = 0,

    pub fn run(self: Markov, allocator: std.mem.Allocator, cfg: Config) !Report {
        const json = try std.fmt.allocPrint(
            allocator,
            "{{\"family\":\"markov\",\"k\":{d},\"start\":{d},\"base_label\":{d},\"theta_max\":{d}}}",
            .{ self.k, self.start, self.base_label, self.theta_max },
        );
        defer allocator.free(json);
        return runFamily(allocator, json, cfg);
    }
};

pub fn version() []const u8 {
    return std.mem.span(c.pk_version());
}

pub fn schemaVersion() []const u8 {
    return std.mem.span(c.pk_schema_version());
}

/// Host vector path: "scalar", "neon" or "avx2". Informational; it never
/// changes a computed value.
pub fn simdPath() []const u8 {
    return std.mem.span(c.pk_simd_path());
}

pub fn gpuAvailable() bool {
    return c.pk_gpu_available() != 0;
}
