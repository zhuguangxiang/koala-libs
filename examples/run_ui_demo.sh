#!/bin/bash
#
# run_ui_demo.sh — 运行 koala/ui 桌面 app 示例（零 HTML/JS/CSS）。
#
# 在任意目录执行均可：脚本按自身位置定位 examples/ 并设置库/包搜索路径。
# 需先构建 koala/ui（在 koala-libs 里 wombat build koala/ui，会带上
# koala/webview、koala/json）以及 koala/datetime（wombat build koala/datetime）。
#
#     ./examples/run_ui_demo.sh
#
# 关闭窗口即退出。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# examples/ lives inside the koala-libs checkout; std comes from the
# installed toolchain.
KOALA_LIBS_ROOT="${KOALA_LIBS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export LD_LIBRARY_PATH=$KOALA_LIBS_ROOT/koala/dist/koala:$HOME/.local/lib:$LD_LIBRARY_PATH
export KOALA_PATH=$KOALA_LIBS_ROOT/koala/dist

exec koala "$SCRIPT_DIR/ui_demo.kl"