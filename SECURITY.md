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
| crates.io | none | trusted publishing over OIDC |
| PyPI | `PYPI_API_TOKEN` | GitHub Actions secret, `pypi` environment |

crates.io needs no stored credential. `release.yml` exchanges the
workflow's OIDC token for a short-lived registry token through
`rust-lang/crates-io-auth-action`, and crates.io only honours it for
this repository, this workflow file and the `crates-io` environment.

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
crates.io is already done: the publisher is configured and the token
that made the first publish has been revoked.

Until PyPI follows, scope its token as narrowly as the registry allows.
The first publish needed an account-scoped token because the project
did not exist yet; now that it does, a project-scoped one is enough.

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
