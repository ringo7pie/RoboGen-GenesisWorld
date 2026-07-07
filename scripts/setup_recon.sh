#!/bin/bash
# ============================================================
# setup_recon.sh
# 役割: ノートブック③（動画/画像 → 3DGS + 物理付き USD）用の環境を Colab 上に構築する
#   ① Python 依存のインストール（requirements/recon.txt）
#   ② gsplat リポジトリの clone（examples/simple_trainer.py を使うため。タグ固定）
#   ③ rmbrualla/pycolmap パーサの隔離インストール
#      （gsplat examples が要求。公式 pycolmap と同名モジュールのため --target で分離し、
#        学習サブプロセスの PYTHONPATH でのみ注入する）
# 引数: なし（配置先は環境変数 GSPLAT_DIR / GSPLAT_PARSER_DIR で変更可）
# 所要: 5〜10 分（fused-ssim の CUDA ビルド含む）
# ============================================================
set -e   # エラー時は即座に停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"       # このスクリプトの所在ディレクトリ
REPO_DIR="$(dirname "$SCRIPT_DIR")"               # 本リポジトリのルート
GSPLAT_DIR="${GSPLAT_DIR:-/content/gsplat}"       # gsplat リポジトリの clone 先
GSPLAT_PARSER_DIR="${GSPLAT_PARSER_DIR:-/content/gsplat_parser}"   # パーサの隔離インストール先
GSPLAT_TAG="v1.5.3"                               # gsplat のバージョン（requirements の gsplat== と一致させる）
PARSER_COMMIT="cc7ea4b7301720ac29287dbe450952511b32125e"   # rmbrualla/pycolmap のコミット（gsplat examples 記載値）

echo "=== [1/4] Python バージョン確認 ==="
python3 - <<'EOF'
import sys                                        # バージョン判定用
v = sys.version_info                              # 実行中の Python バージョン
print(f"Python {v.major}.{v.minor}.{v.micro}")
assert (3, 10) <= (v.major, v.minor) < (3, 14), \
    f"Python 3.10〜3.13 が必要です（現在: {v.major}.{v.minor}）"
EOF

echo "=== [2/4] Python パッケージのインストール（数分。fused-ssim の CUDA ビルドを含む） ==="
# torch / numpy / scipy / opencv は Colab プリインストールを使う（recon.txt に含めない）
pip install -q -r "$REPO_DIR/requirements/recon.txt"
# ArUco（スケール校正）は opencv-contrib に含まれる。Colab 標準 cv2 に無い場合のみ導入する
python3 -c "import cv2; cv2.aruco.ArucoDetector" 2>/dev/null || pip install -q opencv-contrib-python

echo "=== [3/4] gsplat リポジトリの clone（$GSPLAT_TAG 固定） ==="
if [ ! -d "$GSPLAT_DIR/.git" ]; then
    git clone -q --depth 1 --branch "$GSPLAT_TAG" https://github.com/nerfstudio-project/gsplat.git "$GSPLAT_DIR"
fi

echo "=== [4/4] COLMAP パーサ（rmbrualla/pycolmap）の隔離インストール ==="
# 公式 pycolmap（SfM 用）と同名のため site-packages には入れず、専用ディレクトリに隔離する。
# train_gsplat.py が学習サブプロセスの PYTHONPATH 先頭にこのディレクトリを注入する。
if [ ! -d "$GSPLAT_PARSER_DIR/pycolmap" ]; then
    pip install -q --target "$GSPLAT_PARSER_DIR" --no-deps \
        "git+https://github.com/rmbrualla/pycolmap@$PARSER_COMMIT"
fi

echo ""
echo "セットアップ完了。次のセル（check_env.py --stage recon）で環境を検証してください。"
echo "（初回はそこで gsplat の CUDA カーネル JIT コンパイルが 1〜3 分走ります）"
