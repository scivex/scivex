#!/usr/bin/env bash
# Sync pyscivex version with workspace Cargo.toml version.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
sed -i.bak "s/^version = .*/version = \"$VERSION\"/" crates/pyscivex/pyproject.toml
rm -f crates/pyscivex/pyproject.toml.bak
echo "Synced pyscivex to v$VERSION"
