// wasm-pack writes pkg/.gitignore containing "*". npm honours nested
// ignore files, so without this the published tarball has no wasm in it.
import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
rmSync(fileURLToPath(new URL("../pkg/.gitignore", import.meta.url)), { force: true });
