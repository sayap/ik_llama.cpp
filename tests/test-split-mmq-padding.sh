#!/bin/sh
# Runs test-split-mmq-padding under compute-sanitizer memcheck.
#
# This is the memory-level regression check for the generic split buffer
# (ggml_backend_split_buffer_type) ignoring per-backend quantized-row padding:
# before the fix the CUDA MMQ kernels read past the end of unpadded slice
# allocations (an "illegal memory access" whenever the tail bordered an
# unmapped page); memcheck flags the reads deterministically.
#
# Usage: test-split-mmq-padding.sh <path-to-test-binary> [args...]
#
# Exit codes:
#   0                clean (or skipped: no sanitizer / no CUDA device)
#   99               compute-sanitizer reported memory errors
#   anything else    the test binary itself failed
set -u

BIN=${1:?usage: test-split-mmq-padding.sh <test-binary> [args...]}
shift || true

# locate compute-sanitizer
SAN=""
if command -v compute-sanitizer >/dev/null 2>&1; then
    SAN=$(command -v compute-sanitizer)
else
    for d in /usr/local/cuda/bin /usr/local/cuda*/bin /opt/cuda/bin; do
        if [ -x "$d/compute-sanitizer" ]; then
            SAN="$d/compute-sanitizer"
            break
        fi
    done
fi
if [ -z "$SAN" ]; then
    echo "test-split-mmq-padding-sanitizer: SKIP (compute-sanitizer not found)"
    exit 0
fi

echo "using $SAN"
"$SAN" --tool memcheck --error-exitcode 99 "$BIN" "$@"
rc=$?
if [ "$rc" -eq 99 ]; then
    echo "test-split-mmq-padding-sanitizer: FAIL (compute-sanitizer reported memory errors)"
    exit 99
fi
if [ "$rc" -ne 0 ]; then
    echo "test-split-mmq-padding-sanitizer: FAIL (test exit code $rc)"
    exit "$rc"
fi
echo "test-split-mmq-padding-sanitizer: PASS"
exit 0
