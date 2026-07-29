//! Standalone conformance check for the Zig binding.
//!
//! Built as an object and linked with the system linker, which keeps it
//! portable across the platforms CI covers. Exits non-zero on failure.

const std = @import("std");
const pk = @import("perturbation_kernel.zig");

var failures: u32 = 0;

fn check(ok: bool, what: []const u8) void {
    std.debug.print("  {s:<58} {s}\n", .{ what, if (ok) "ok" else "FAILED" });
    if (!ok) failures += 1;
}

pub fn main() !u8 {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    std.debug.print("perturbation-kernel {s} (schema {s}), simd path {s}, gpu {s}\n\n", .{
        pk.version(), pk.schemaVersion(), pk.simdPath(),
        if (pk.gpuAvailable()) "yes" else "no",
    });

    {
        var r = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
            .run(a, .{ .n = 262144, .seed = 20260610 });
        defer r.deinit();
        check(r.value() == 0.8802871704101562, "markov matches the reference value exactly");
        check(std.mem.indexOf(u8, r.json(), "tail_survival") != null,
            "report json carries the functional tag");
    }
    {
        var s = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
            .run(a, .{ .n = 262144, .seed = 20260610, .backend = .scalar });
        defer s.deinit();
        var v = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
            .run(a, .{ .n = 262144, .seed = 20260610, .backend = .simd });
        defer v.deinit();
        check(s.value() == v.value(), "scalar and simd agree bit for bit");
    }
    {
        var m = try (pk.Markov{ .k = 5, .theta_max = 0.0, .start = 2, .base_label = 2 })
            .run(a, .{ .n = 10000, .seed = 5 });
        defer m.deinit();
        check(m.value() == 1.0, "null intensity recovers the base label exactly");

        const base = [_]f64{ 1.5, -2.0 };
        var g = try (pk.Gaussian{ .base = &base, .sigma_max = 0.0 })
            .run(a, .{ .n = 10000, .seed = 5 });
        defer g.deinit();
        check(g.value() == 0.0, "null intensity gives zero dispersion");
    }
    {
        var b = try (pk.Bistable{ .x0 = 0.0, .dt = 0.01, .theta_max = 0.5 })
            .run(a, .{ .n = 20000, .seed = 3 });
        defer b.deinit();
        check(b.value() >= -1.0 and b.value() <= 1.0, "polarisation lies in [-1, 1]");
    }
    {
        const e1 = (pk.Markov{ .k = 5, .theta_max = 0.3 }).run(a, .{ .n = 0, .seed = 1 });
        check(e1 == pk.Error.EmptyEnsemble, "an empty ensemble is an error");
        const e2 = (pk.Markov{ .k = 5, .theta_max = 0.3 }).run(a, .{
            .n = 1000, .seed = 1, .invariance_lambda = 1.0,
            .epsilon = 0.05, .eta = 0.05, .observation_diameter = 1.0, .obs_dim = 1,
        });
        check(e2 == pk.Error.SampleFloor, "an unsupported accuracy claim is an error");
        const e3 = (pk.Markov{ .k = 0, .theta_max = 0.3 }).run(a, .{ .n = 16, .seed = 1 });
        check(e3 == pk.Error.InvalidConfig, "an out-of-domain family is an error");
    }
    if (pk.gpuAvailable()) {
        var h = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
            .run(a, .{ .n = 262144, .seed = 20260610, .backend = .scalar });
        defer h.deinit();
        var d = try (pk.Markov{ .k = 5, .theta_max = 0.3 })
            .run(a, .{ .n = 262144, .seed = 20260610, .backend = .gpu });
        defer d.deinit();
        check(h.value() == d.value(), "gpu is bit-identical to the host");
    } else {
        std.debug.print("  {s:<58} skipped\n", .{"gpu (no device on this machine)"});
    }

    std.debug.print("\n{s}\n", .{if (failures == 0) "all checks passed" else "FAILURES"});
    return if (failures == 0) 0 else 1;
}
