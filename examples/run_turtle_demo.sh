#!/bin/bash
#
# run_turtle_demo.sh — 运行 koala/turtle 彩色螺旋示例。
#
# 在任意目录执行均可：脚本按自身位置定位 examples/ 并设置库/包搜索路径。
# 需先构建 koala/turtle（在 koala-libs 里 wombat build koala/turtle）和 SDL2。
#
#     ./examples/run_turtle_demo.sh
#
# 关闭窗口（或按 Esc）即退出。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# examples/ lives inside the koala-libs checkout; std comes from the
# installed toolchain.
KOALA_LIBS_ROOT="${KOALA_LIBS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export LD_LIBRARY_PATH=$KOALA_LIBS_ROOT/koala/dist/koala:$HOME/.local/lib:$LD_LIBRARY_PATH
export KOALA_PATH=$KOALA_LIBS_ROOT/koala/dist

exec koala "$SCRIPT_DIR/turtle_demo.kl"