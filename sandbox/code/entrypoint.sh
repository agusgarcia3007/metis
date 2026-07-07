#!/bin/sh
set -eu

fail_setup() {
  echo "METIS_SANDBOX_SETUP_ERROR: $*" >&2
  exit 70
}

cp -a /input/. /workspace/ || fail_setup "could not copy the input repository"
git -C /workspace apply --whitespace=nowarn /candidate/candidate.patch \
  || fail_setup "candidate is not a valid applicable patch"

# Held-out files are verifier-owned and arrive only after the model's patch is fixed.
if [ -d /held-out ]; then
  cp -a /held-out/. /workspace/ || fail_setup "could not inject held-out tests"
fi

# The pilot toolchain is available offline. A project-provided node_modules takes precedence; tiny
# benchmark fixtures can use the image's pinned toolchain without vendoring dependencies.
if [ ! -e /workspace/node_modules ]; then
  ln -s /opt/metis-tools/node_modules /workspace/node_modules \
    || fail_setup "could not expose the offline TypeScript toolchain"
fi

# Candidate code gets a readable but immutable workspace. Test-generated files belong in /tmp.
chmod -R a-w /workspace || fail_setup "could not seal the verification workspace"

cd /workspace
exec /usr/sbin/gosu node "$@"
