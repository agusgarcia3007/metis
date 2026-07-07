# Phase 5 TypeScript code sandbox

This image is the execution boundary for `VerifierKind::Exec`. Build it once:

```sh
docker build -t metis-code-sandbox:phase5 sandbox/code
```

At runtime Metis starts one fresh container per gate with:

- no network;
- a read-only root filesystem and read-only host mounts;
- 1 CPU, 1 GiB RAM, 256 PIDs, a 120-second timeout, and bounded captured output;
- all capabilities dropped except `SETUID`/`SETGID`, used only by the entrypoint to become UID 1000;
- the patch applied before held-out tests are injected;
- the resulting workspace sealed read-only before candidate code executes.

The pinned pilot toolchain exposes four argv-only commands: `metis-ts-parse`,
`metis-ts-typecheck`, `metis-ts-lint`, and `metis-vitest`. A project-local `node_modules` takes
precedence; otherwise the image's offline dependencies are used.

The host-side patch policy also rejects edits to tests, package/tooling configuration, lockfiles,
Docker files, `.git`, and `node_modules`, plus newly-added skip/only/type-suppression directives.
Docker is still a shared-kernel boundary; production deployment should keep the runtime and kernel
patched and may replace it with a stronger VM sandbox without changing the verifier contract.

## H2 smoke

```sh
cargo run --release --bin h2 -- \
  bench/h2-smoke/dataset.json \
  bench/results-h2-smoke.json
```

The smoke set validates the harness and oracle plumbing. It is intentionally labeled as hand-authored
and is not the preregistered ~20-task SWE-bench experiment from the design plan.
