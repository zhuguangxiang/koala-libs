#!/bin/bash
#
# C unit test for the native core of koala/re (re_core.c + re_unicode.c).
#
# Depends only on the installed Koala toolchain: utf8.h comes from
# <prefix>/include/koala/common and the utf8 helpers are linked from
# <prefix>/lib/libkoala_common.a. The prefix is the installation that owns the
# `koala` executable in PATH; set KOALA_PREFIX to use another one.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "${KOALA_PREFIX:-}" ]; then
    KOALA_BIN="$(command -v koala || true)"
    if [ -z "$KOALA_BIN" ]; then
        echo "test_re_core: koala not found in PATH; set KOALA_PREFIX explicitly" >&2
        exit 1
    fi
    # <prefix>/bin/koala -> <prefix>
    KOALA_PREFIX="$(dirname "$(dirname "$(readlink -f "$KOALA_BIN")")")"
fi

KOALA_INC="$KOALA_PREFIX/include/koala/common"
KOALA_LIB="$KOALA_PREFIX/lib"

if [ ! -f "$KOALA_INC/utf8.h" ]; then
    echo "test_re_core: missing $KOALA_INC/utf8.h; install the koala toolchain first" >&2
    exit 1
fi

if [ ! -f "$KOALA_LIB/libkoala_common.a" ]; then
    echo "test_re_core: missing $KOALA_LIB/libkoala_common.a; set KOALA_PREFIX" >&2
    exit 1
fi

echo "=============================================="
echo " test_re_core: koala/re native C unit test"
echo "=============================================="

gcc -std=c23 -g -O0 \
    -I"$NATIVE_DIR" -I"$KOALA_INC" \
    "$SCRIPT_DIR/test_re_core.c" \
    "$NATIVE_DIR/re_core.c" \
    "$NATIVE_DIR/re_unicode.c" \
    "$KOALA_LIB/libkoala_common.a" \
    -lm -o "$SCRIPT_DIR/test_re_core"

"$SCRIPT_DIR/test_re_core"