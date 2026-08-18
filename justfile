# SPDX-License-Identifier: Apache-2.0
# Run `just` to see available recipes.

# List available recipes.
default:
    @just --list

# Run every blocking CI check locally before pushing (fmt, clippy, tests, packaging).
ci: fmt-check lint test-ci package

# Check formatting without modifying files (matches the CI `fmt` job).
fmt-check:
    cargo fmt --all -- --check

# Apply rustfmt formatting in place.
fmt:
    cargo fmt --all

# Lint with the exact flags the CI `clippy` job enforces (pedantic + warnings + unused_must_use as errors).
lint:
    cargo clippy --all-targets --all-features --workspace -- -W clippy::pedantic -D warnings -W unused_must_use

# Run the full test suite (default features).
test:
    cargo test

# Run tests exactly as the CI `test` job does (all features + doctests).
test-ci:
    cargo test --all-features
    cargo test --doc --all-features

# Run tests across feature configurations (default + nvidia disabled).
test-all:
    cargo test
    cargo test --no-default-features

# Verify the crate packages cleanly and report its size (matches CI; --allow-dirty so it runs pre-commit).
package:
    cargo publish --dry-run --allow-dirty

# Measure code coverage (requires cargo-llvm-cov). --remap-path-prefix keeps the
# report paths relative (src/...), and tests/ is excluded so only product code is
# counted. No --locked: this is a library, so Cargo.lock is deliberately not
# committed. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --all-features --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --all-features --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'
