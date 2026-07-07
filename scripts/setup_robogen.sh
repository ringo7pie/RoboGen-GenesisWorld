#!/bin/bash
# ============================================================
# setup_robogen.sh
# 役割: ノートブック②（RoboGen）用の環境を Colab 上に一括構築する
#       ① RoboGen をコミット固定で clone
#       ② Python 依存パッケージのインストール
#       ③ 置き換えモジュール（gym shim / SB3 学習部）のコピー
#       ④ パッチ適用
# 引数: なし（配置先は環境変数 ROBOGEN_DIR で変更可、既定 /content/RoboGen）
# 所要: 5〜10 分
# ============================================================
set -e   # エラー時は即座に停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"     # このスクリプトの所在ディレクトリ
REPO_DIR="$(dirname "$SCRIPT_DIR")"             # 本リポジトリのルート
ROBOGEN_DIR="${ROBOGEN_DIR:-/content/RoboGen}"  # RoboGen の clone 先
ROBOGEN_COMMIT="486612cd0baa40857b561419c5a42fbb7e67724f"   # 動作確認済みコミット（2024-05-30、パッチの適用先として固定）

echo "=== [1/5] Python バージョン確認 ==="
python3 - <<'EOF'
import sys                                       # バージョン判定用
v = sys.version_info                             # 実行中の Python バージョン
print(f"Python {v.major}.{v.minor}.{v.micro}")
assert (3, 10) <= (v.major, v.minor) < (3, 14), \
    f"Python 3.10〜3.13 が必要です（現在: {v.major}.{v.minor}）"
EOF

echo "=== [2/5] RoboGen の clone（コミット固定） ==="
if [ ! -d "$ROBOGEN_DIR/.git" ]; then
    git clone https://github.com/Genesis-Embodied-AI/RoboGen.git "$ROBOGEN_DIR"
fi
cd "$ROBOGEN_DIR"
git checkout -q "$ROBOGEN_COMMIT"                # パッチ適用先を固定するためコミットを固定

echo "=== [3/5] Python パッケージのインストール（数分かかります） ==="
# torch / numpy / scipy / pandas は Colab プリインストールを使う（requirements に含めない）
pip install -q -r "$REPO_DIR/requirements/robogen.txt"

echo "=== [4/5] 置き換えモジュールのコピー ==="
# gym→gymnasium shim: RoboGen 直下に置くことで site-packages より優先される
cp -r "$REPO_DIR/robogen_overrides/gym" "$ROBOGEN_DIR/"
# SB3 版 RL 学習部: ray 1.13 ベースの ray_learn.py の代替
cp "$REPO_DIR/robogen_overrides/RL/sb3_learn.py" "$ROBOGEN_DIR/RL/"

echo "=== [5/5] パッチ適用 ==="
bash "$REPO_DIR/scripts/apply_patches.sh" "$ROBOGEN_DIR"

echo ""
echo "セットアップ完了。次のセル（check_env.py --stage robogen）で環境を検証してください。"
echo "注意: RoboGen のスクリプトは必ず $ROBOGEN_DIR を作業ディレクトリとして実行してください（相対パス前提のため）。"
