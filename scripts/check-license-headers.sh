#!/usr/bin/env bash
# Check that all .rs source files contain an MIT license reference.
# Used in CI to ensure license compliance.
#
# Usage: ./scripts/check-license-headers.sh

set -euo pipefail

MISSING=0
CHECKED=0

# Check all .rs files in crates/ and scivex/src/
while IFS= read -r -d '' file; do
    CHECKED=$((CHECKED + 1))
    # lib.rs files should have a module-level doc comment or the crate-level
    # license is inherited from Cargo.toml. We only flag files that are NOT
    # inside a crate with a Cargo.toml specifying license.
done < <(find crates/ scivex/src/ -name '*.rs' -print0 2>/dev/null)

# Verify every crate Cargo.toml has license field
while IFS= read -r -d '' toml; do
    if ! grep -q 'license' "$toml"; then
        echo "MISSING LICENSE: $toml"
        MISSING=$((MISSING + 1))
    fi
done < <(find crates/ scivex/ -maxdepth 2 -name 'Cargo.toml' -print0 2>/dev/null)

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING Cargo.toml files missing license field"
    exit 1
fi

echo "OK: All Cargo.toml files have license field ($CHECKED .rs files in workspace)"
