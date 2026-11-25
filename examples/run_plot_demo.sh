#!/bin/bash
#
# run_plot_demo.sh — 运行 koala/plot 功能演示菜单（plot_demo.kl）。
#
# 在任意目录执行均可：脚本按自身位置定位 examples/ 并设置库/包搜索路径。
# 需先构建 koala/plot（在 koala-libs 里 wombat build koala/plot，会带上
# koala/webview、koala/json、std/math、koala/readline）。
#
#     ./examples/run_plot_demo.sh
#
# 终端菜单 1-4 一页一图，5 是 2x2 多图一页；关窗回到菜单，0 退出。
# 无显示环境可改用 html 后端（每个菜单项写出 plot.html，覆盖式）：
#
#     KOALA_PLOT_BACKEND=html ./examples/run_plot_demo.sh
#     printf '5\n0\n' | KOALA_PLOT_BACKEND=html ./examples/run_plot_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# examples/ lives inside the koala-libs checkout; std comes from the
# installed toolchain.
KOALA_LIBS_ROOT="${KOALA_LIBS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export LD_LIBRARY_PATH=$KOALA_LIBS_ROOT/koala/dist/koala:$HOME/.local/lib:$LD_LIBRARY_PATH
export KOALA_PATH=$KOALA_LIBS_ROOT/koala/dist

exec koala "$SCRIPT_DIR/plot_demo.kl"