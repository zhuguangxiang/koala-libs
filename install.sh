#!/bin/bash
#
# Install the built koala/* packages into the installed Koala toolchain.
#
# The destination is the `lib/koala-packages/` directory of the installation
# that owns the `koala` executable in PATH. The standard library lives one
# level up, in `lib/std/`. Set KOALA_LIB_DIR to install somewhere else.
#
# Build first:  ./koala/build-and-test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="$SCRIPT_DIR/koala/dist"

if [ ! -d "$DIST" ]; then
    echo "install: missing $DIST; run './koala/build-and-test.sh' first" >&2
    exit 1
fi

if [ -z "${KOALA_LIB_DIR:-}" ]; then
    KOALA_BIN="$(command -v koala || true)"
    if [ -z "$KOALA_BIN" ]; then
        echo "install: koala not found in PATH; set KOALA_LIB_DIR explicitly" >&2
        exit 1
    fi
    # <prefix>/bin/koala -> <prefix>/lib/koala-packages
    PREFIX="$(dirname "$(dirname "$(readlink -f "$KOALA_BIN")")")"
    KOALA_LIB_DIR="$PREFIX/lib/koala-packages"
fi

echo "=============================================="
echo " install: koala/* -> $KOALA_LIB_DIR"
echo "=============================================="

mkdir -p "$KOALA_LIB_DIR"
cp -a "$DIST/." "$KOALA_LIB_DIR/"

echo "installed $(find "$DIST" -type f | wc -l) file(s) from $DIST"
echo ""
echo "native libraries live in $KOALA_LIB_DIR/koala;"
echo "put that directory on LD_LIBRARY_PATH to load them."