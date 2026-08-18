#!/usr/bin/env bash
# Downloads a real tekken.json for every supported tokenizer version, used by
# tests/test_real_tokenizers.rs.
#
# Each file is fetched from a pinned revision of a public Mistral model on
# Hugging Face (no auth required) and verified against its sha256. Files are
# cached at tests/assets/tekken_v<N>.json (gitignored); re-running is a no-op
# for files that already match their checksum.
set -euo pipefail

ASSETS_DIR="$(dirname "$0")/../tests/assets"

# version|repo|revision|sha256
ASSETS=(
    "v3|mistralai/Mistral-Nemo-Instruct-2407|04d8a90549d23fc6bd7f642064003592df51e9b3|eccd1665d2e477697c33cb7f0daa6f6dfefc57a0a6bceb66d4be52952f827516"
    "v7|mistralai/Mistral-Small-24B-Instruct-2501|9527884be6e5616bdd54de542f9ae13384489724|c4b90a968dbc67ef3975129d0b78a2e3cbb6bea340ab9205f22e8a0308b1ffc5"
    "v11|mistralai/Mistral-Small-3.2-24B-Instruct-2506|95a6d26c4bfb886c58daf9d3f7332c857cb27b43|6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1"
    "v13|mistralai/Ministral-3-8B-Instruct-2512|5b26027e7b19eeb4b7352e1fed3926375dd2cb4d|600bb27946565481ecf51ba8aee252e49b9a68507866080ac9c30185bb312843"
    "v15|mistralai/Mistral-Small-4-119B-2603|a11f36bebf709121056b1dbcc943d1c6afbe494d|b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7"
)

failures=0
for entry in "${ASSETS[@]}"; do
    IFS='|' read -r version repo revision sha256 <<< "${entry}"
    dest="${ASSETS_DIR}/tekken_${version}.json"
    url="https://huggingface.co/${repo}/resolve/${revision}/tekken.json"

    if [ -f "${dest}" ] && echo "${sha256}  ${dest}" | sha256sum --check --quiet - 2>/dev/null; then
        echo "✓ ${dest} already present with correct checksum"
        continue
    fi

    echo "Downloading ${version} tokenizer from ${repo}..."
    if ! curl --fail --location --retry 3 --silent --show-error --output "${dest}.tmp" "${url}"; then
        echo "✗ Download failed for ${version} (${url})" >&2
        rm -f "${dest}.tmp"
        failures=$((failures + 1))
        continue
    fi
    mv "${dest}.tmp" "${dest}"

    if ! echo "${sha256}  ${dest}" | sha256sum --check --quiet -; then
        echo "✗ Checksum mismatch for ${dest}" >&2
        rm -f "${dest}"
        failures=$((failures + 1))
        continue
    fi
    echo "✓ Downloaded and verified ${dest}"
done

exit "${failures}"
