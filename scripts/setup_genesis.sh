#!/bin/bash
# ============================================================
# setup_genesis.sh
# 役割: ノートブック①（Genesis 単体デモ）用の環境を Colab 上に構築する
# 引数: なし
# 前提: Colab (Ubuntu, Python 3.10〜3.13, GPU ランタイム) で実行
# ============================================================
set -e   # エラー時は即座に停止（途中成功のまま進まないため）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # このスクリプトの所在ディレクトリ
REPO_DIR="$(dirname "$SCRIPT_DIR")"           # リポジトリのルート（scripts/ の親）

echo "=== [1/3] Python バージョン確認 ==="
# genesis-world は Python >=3.10, <3.14 が必要。範囲外なら警告して停止する
python3 - <<'EOF'
import sys                                        # バージョン判定用
v = sys.version_info                              # 実行中の Python バージョン
print(f"Python {v.major}.{v.minor}.{v.micro}")
assert (3, 10) <= (v.major, v.minor) < (3, 14), \
    f"genesis-world は Python 3.10〜3.13 が必要です（現在: {v.major}.{v.minor}）"
EOF

echo "=== [2/3] OS パッケージ確認（EGL / ffmpeg） ==="
# ヘッドレスレンダリング用の EGL と動画変換用の ffmpeg。Colab には通常入っているが念のため
if ! dpkg -s libegl1 >/dev/null 2>&1; then
    apt-get -qq update && apt-get -qq install -y libegl1   # EGL ランタイム（オフスクリーン描画に必要）
fi
command -v ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)

echo "=== [3/3] Python パッケージのインストール ==="
# torch / numpy は Colab プリインストールを使うため requirements には含めていない
pip install -q -r "$REPO_DIR/requirements/genesis.txt"

echo ""
echo "セットアップ完了。次のセル（check_env.py --stage genesis）で環境を検証してください。"
