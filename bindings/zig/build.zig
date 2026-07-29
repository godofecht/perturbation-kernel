//! Build for the Zig binding.
//!
//! Expects the Rust static library to exist already:
//!
//!     cargo build --release --all-features
//!     zig build check
//!
//! `-Drust-target-dir=` overrides where to look for it.
//!
//! On macOS, Zig 0.14's Mach-O reader rejects archive members produced
//! by current Apple toolchains, so the link step fails there through no
//! fault of the binding. The two-step form works, and is what the
//! macOS CI job runs:
//!
//!     zig build-obj check.zig -I ../cpp/include -lc -femit-bin=check.o
//!     cc check.o ../../target/release/libperturbation_kernel.a \
//!        -framework CoreFoundation -framework Foundation -framework Metal \
//!        -framework QuartzCore -framework CoreGraphics -framework IOKit \
//!        -framework IOSurface -framework AppKit -lobjc -liconv -o check

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rust_dir = b.option(
        []const u8,
        "rust-target-dir",
        "Directory holding the compiled Rust static library",
    ) orelse "../../target/release";

    const exe = b.addExecutable(.{
        .name = "check",
        .root_source_file = b.path("check.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.addIncludePath(b.path("../cpp/include"));

    // Link the static archive explicitly rather than through
    // `linkSystemLibrary`, which would prefer the dylib cargo also
    // produces.
    const lib_name = if (target.result.os.tag == .windows)
        "perturbation_kernel.lib"
    else
        "libperturbation_kernel.a";
    exe.addObjectFile(.{ .cwd_relative = b.pathJoin(&.{ rust_dir, lib_name }) });
    exe.linkLibC();

    // The Rust staticlib pulls in the platform's system libraries; wgpu
    // adds the graphics frameworks when the gpu feature is on.
    switch (target.result.os.tag) {
        .macos => {
            for ([_][]const u8{
                "CoreFoundation", "Foundation", "Metal",     "QuartzCore",
                "CoreGraphics",   "IOKit",      "IOSurface", "AppKit",
            }) |fw| exe.linkFramework(fw);
            exe.linkSystemLibrary("objc");
            exe.linkSystemLibrary("iconv");
        },
        .windows => {
            for ([_][]const u8{
                "ntdll", "userenv", "ws2_32", "bcrypt", "advapi32",
            }) |l| exe.linkSystemLibrary(l);
        },
        else => {
            exe.linkSystemLibrary("pthread");
            exe.linkSystemLibrary("dl");
            exe.linkSystemLibrary("m");
            // The Rust staticlib references the unwinder for panics.
            // Zig's lld does not pull it in implicitly the way cc does.
            exe.linkSystemLibrary("gcc_s");
        },
    }

    b.installArtifact(exe);
    b.step("check", "Run the binding conformance checks")
        .dependOn(&b.addRunArtifact(exe).step);
}
