#!/bin/bash
# ============================================================
# apply_patches.sh
# 役割: patches/*.patch を RoboGen リポジトリへ順番に適用する（冪等）
#       既に適用済みのパッチは検出してスキップするため、セルの再実行に耐える
# 引数: $1 = RoboGen のディレクトリ（省略時 /content/RoboGen）
# ============================================================
set -e   # エラー時は即座に停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"     # このスクリプトの所在ディレクトリ
REPO_DIR="$(dirname "$SCRIPT_DIR")"             # 本リポジトリのルート
ROBOGEN_DIR="${1:-/content/RoboGen}"            # パッチ適用先の RoboGen ディレクトリ

cd "$ROBOGEN_DIR"

APPLIED_LIST=""                                  # 適用済みパッチ名の記録用
for patch_file in "$REPO_DIR"/patches/*.patch; do
    name="$(basename "$patch_file")"             # パッチファイル名（表示用）
    if git apply --check "$patch_file" 2>/dev/null; then
        git apply "$patch_file"                  # 未適用 → 適用する
        echo "[適用] $name"
    elif git apply --reverse --check "$patch_file" 2>/dev/null; then
        echo "[スキップ] $name（適用済み）"       # 逆適用が通る = 既に当たっている
    else
        echo "[エラー] $name が適用できません。RoboGen のコミットが想定と異なる可能性があります。" >&2
        exit 1
    fi
    APPLIED_LIST="$APPLIED_LIST $name"
done

# 適用済みマーカーを書き出す（check_env.py --stage robogen が検査する）
COMMIT_HASH="$(git rev-parse HEAD)"              # 現在の RoboGen コミット
echo "commit=$COMMIT_HASH patches=$APPLIED_LIST" > .patches_applied
echo "パッチ適用完了: $APPLIED_LIST"
