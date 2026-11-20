// Fails if the npm tarball would be missing a file the package needs
// at runtime. Run before every publish.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const cwd = fileURLToPath(new URL("..", import.meta.url));
const [info] = JSON.parse(
  execFileSync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], { cwd, encoding: "utf8" }),
);
const packed = new Set(info.files.map((f) => f.path));
const required = [
  "index.js",
  "index.d.ts",
  "README.md",
  "pkg/perturbation_kernel_wasm.js",
  "pkg/perturbation_kernel_wasm_bg.wasm",
];
const missing = required.filter((f) => !packed.has(f));
if (missing.length) {
  console.error(`tarball is missing: ${missing.join(", ")}`);
  process.exit(1);
}
console.log(`tarball ok: ${info.files.length} files, ${info.size} bytes`);
