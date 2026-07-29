# Security

## Reporting a vulnerability

Open a [security advisory](https://github.com/godofecht/perturbation-kernel/security/advisories/new),
or email abhishek.shivakumar@gmail.com. Please do not open a public
issue for anything exploitable.

## How releases are published

Releases are cut by pushing a `v*` tag. Nothing is published from a
workstation.

| Registry | Credential | Where it lives |
|---|---|---|
| PyPI | `PYPI_API_TOKEN` | GitHub Actions secret, `pypi` environment |
| crates.io | `CARGO_REGISTRY_TOKEN` | GitHub Actions secret, `crates-io` environment |

Both publishing environments carry a deployment branch policy of `v*`
(tags only). A push to a branch cannot reach them even by triggering
the workflow manually.

`release.yml` is the only workflow that references a secret, and it
triggers on tag pushes and manual dispatch. It never triggers on
`pull_request`, so a pull request from a fork cannot reach a publishing
credential.

Everything else runs with `permissions: contents: read`. The one
exception is `bench.yml`, which needs `contents: write` to commit the
measured numbers back to `BENCHMARKS.md`; it triggers only on pushes to
the default branch, on a schedule, and on manual dispatch.

## Moving to trusted publishing

Tokens are the fallback, not the goal. Both registries support OpenID
Connect, which lets them verify the workflow directly and removes the
stored credential:

- PyPI: add a publisher at
  <https://pypi.org/manage/project/perturbation-kernel/settings/publishing/>
  with owner `godofecht`, repository `perturbation-kernel`, workflow
  `release.yml`, environment `pypi`. Then delete the `PYPI_API_TOKEN`
  secret and remove the `password:` line from the `pypi` job.
  `id-token: write` is already requested, so nothing else changes.
- crates.io: add a Trusted Publishing config in the crate settings with
  the same details, then delete `CARGO_REGISTRY_TOKEN`.

Until then, scope the tokens as narrowly as the registry allows. A
first publish needs an account-scoped PyPI token because the project
does not exist yet; once it does, replace it with a project-scoped one.

## What the library itself touches

Nothing. It opens no sockets, reads no environment variables, and
writes no files. It reads no system randomness either: every random bit
comes from a `ChaCha20` stream keyed by the seed in the config, which is
what makes a run reproducible in the first place.

The `gpu` feature loads a graphics driver through `wgpu` and submits
compute shaders. The shaders are compiled into the binary and are not
read from disk at runtime.

## Supply chain

`cargo publish` ships only `src/`, `benches/`, `tests/`, `examples/`,
`GOLDEN.txt` and the manifests. The `python/`, `lean/` and `docs/`
directories are excluded.

Wheels are built by `release.yml` on GitHub-hosted runners and are
smoke-tested before upload on every platform where the builder can run
the artefact it just produced.
