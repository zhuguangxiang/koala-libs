#!/bin/bash
#
# run_turtle_move.sh — 运行 koala/turtle 交互示例：方向键移动小球。
#
# 在任意目录执行均可。需先构建 koala/turtle 和安装 SDL2。
#
#     ./examples/run_turtle_move.sh
#
# 操作：↑ ↓ ← → 移动；关闭窗口或按 Esc 退出。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# examples/ lives inside the koala-libs checkout; std comes from the
# installed toolchain.
KOALA_LIBS_ROOT="${KOALA_LIBS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export LD_LIBRARY_PATH=$KOALA_LIBS_ROOT/koala/dist/koala:$HOME/.local/lib:$LD_LIBRARY_PATH
export KOALA_PATH=$KOALA_LIBS_ROOT/koala/dist

exec koala "$SCRIPT_DIR/turtle_move.kl"