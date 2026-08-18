#!/usr/bin/env bash
# Downloads the real v15 tekken.json used by tests/test_v15_real.rs.
#
# The file is fetched from a pinned revision of mistralai/Mistral-Small-4-119B-2603
# on Hugging Face (public, no auth required) and verified against its sha256.
# It is cached at tests/assets/tekken_v15.json (gitignored); re-running is a no-op
# if the cached file already matches the checksum.
set -euo pipefail

REPO="mistralai/Mistral-Small-4-119B-2603"
REVISION="a11f36bebf709121056b1dbcc943d1c6afbe494d"
SHA256="b1272b956bd6edd2d2c674c76896c7661308c9e723997b0afb55ecb429cb5dc7"
URL="https://huggingface.co/${REPO}/resolve/${REVISION}/tekken.json"

DEST="$(dirname "$0")/../tests/assets/tekken_v15.json"

verify() {
    echo "${SHA256}  ${DEST}" | sha256sum --check --quiet -
}

if [ -f "${DEST}" ] && verify; then
    echo "✓ ${DEST} already present with correct checksum"
    exit 0
fi

echo "Downloading v15 tokenizer from ${URL}..."
curl --fail --location --retry 3 --output "${DEST}.tmp" "${URL}"
mv "${DEST}.tmp" "${DEST}"

if ! verify; then
    echo "✗ Checksum mismatch for ${DEST}" >&2
    rm -f "${DEST}"
    exit 1
fi
echo "✓ Downloaded and verified ${DEST}"
