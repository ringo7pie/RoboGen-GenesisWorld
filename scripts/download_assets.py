# ============================================================
# download_assets.py
# 役割: RoboGen が必要とするアセットの取得と配置検証を行う
#   --download        : PartNet-Mobility（RoboGen 配布のパース済み版）を Google Drive の
#                       公開リンクから gdown で取得し、RoboGen/data/dataset に展開する
#   --with-embeddings : Objaverse 検索用の文埋め込み（ステップB のタスク新規生成で使用）も取得する
#   --verify          : 配置状態の検証のみ行う（ダウンロードしない）
# キャッシュ: --cache-dir（ユーザーの Google Drive 推奨）に zip を保存し、
#             セッション切れ後の再ダウンロードを回避する
# ============================================================
import argparse      # コマンドライン引数の解析用
import os            # パス操作用
import shutil        # ファイルコピー用
import sys           # 終了コード用
import zipfile       # zip 展開用

# RoboGen 公式 README 記載の配布物（Google Drive 公開ファイル ID）
PARTNET_GDRIVE_ID = "1d-1txzcg_ke17NkHKAolXlfDnmPePFc6"    # パース済み PartNet-Mobility データセット
EMBEDDINGS_GDRIVE_ID = "1dFDpG3tlckTUSy7VYdfkNqtfVctpn3T6" # Objaverse 検索用の文埋め込み


# Google Drive から zip を取得する（キャッシュ優先）
# 引数: gdrive_id = 公開ファイル ID、cache_path = キャッシュ先 zip パス
# 戻り値: ローカルの zip パス
def fetch_zip(gdrive_id, cache_path):
    if os.path.isfile(cache_path):
        print(f"キャッシュを使用: {cache_path}")
        return cache_path
    import gdown                                  # Google Drive ダウンローダ（遅延 import）
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    print(f"ダウンロード中: https://drive.google.com/uc?id={gdrive_id}")
    gdown.download(id=gdrive_id, output=cache_path, quiet=False)
    assert os.path.isfile(cache_path), f"ダウンロードに失敗しました: {cache_path}"
    return cache_path


# zip を展開し、期待するフォルダ名になるよう配置する
# 引数: zip_path = 展開元、dest_dir = 最終的に作りたいディレクトリ（例 data/dataset）
def extract_to(zip_path, dest_dir):
    parent = os.path.dirname(dest_dir)            # 展開先の親ディレクトリ
    os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # --- 展開前にディスク空き容量を検査する（展開途中の容量切れは復旧が面倒なため） ---
        need = sum(i.file_size for i in zf.infolist())         # 展開後の合計サイズ（バイト）
        free = shutil.disk_usage(parent).free                  # 展開先の空き容量（バイト）
        margin = 2 * 1024 ** 3                                 # 安全マージン 2GB
        assert free > need + margin, (
            f"ディスク空き容量が不足しています（必要 {need/1024**3:.1f}GB + 余裕2GB、"
            f"空き {free/1024**3:.1f}GB）。不要ファイルを削除してから再実行してください。"
        )
        print(f"展開中: {zip_path} → {parent}（展開後 {need/1024**3:.1f}GB / 空き {free/1024**3:.1f}GB）")
        top_names = {n.split("/")[0] for n in zf.namelist()}   # zip 内のトップレベル名一覧
        zf.extractall(parent)
    # zip のトップレベル名が dest_dir と異なる場合はリネームして揃える
    want = os.path.basename(dest_dir)             # 期待するフォルダ名
    if want not in top_names and len(top_names) == 1:
        actual = os.path.join(parent, top_names.pop())         # 実際に展開されたフォルダ
        if not os.path.isdir(dest_dir):
            os.rename(actual, dest_dir)
    assert os.path.isdir(dest_dir), f"展開結果が想定と異なります: {dest_dir} が存在しません"


# PartNet-Mobility の配置を検証する（mobility.urdf を持つオブジェクト数を数える）
# 戻り値: (成否, メッセージ)
def verify_partnet(robogen_dir):
    dataset_dir = os.path.join(robogen_dir, "data", "dataset")   # データセットの配置先
    if not os.path.isdir(dataset_dir):
        return False, f"{dataset_dir} がありません。--download を実行してください。"
    count = 0                                     # mobility.urdf を持つオブジェクト数
    for entry in os.listdir(dataset_dir):
        if os.path.isfile(os.path.join(dataset_dir, entry, "mobility.urdf")):
            count += 1
    if count == 0:
        return False, f"{dataset_dir} に mobility.urdf を持つオブジェクトがありません（展開失敗の可能性）"
    return True, f"PartNet-Mobility OK（{count} オブジェクト）"


# Objaverse 検索用埋め込みの配置を検証する（ステップB でのみ必要）
# 戻り値: (成否, メッセージ)
def verify_embeddings(robogen_dir):
    data_dir = os.path.join(robogen_dir, "objaverse_utils", "data")   # 埋め込みの配置先
    need = os.path.join(data_dir, "cap3d_sentence_bert_embeddings.pt")   # 代表ファイル
    if not os.path.isfile(need):
        return False, f"{need} がありません（新規オブジェクト検索を使う場合は --download --with-embeddings を実行）"
    return True, "Objaverse 埋め込み OK"


def main():
    parser = argparse.ArgumentParser(description="RoboGen アセットの取得・検証")
    parser.add_argument("--robogen-dir", default=os.environ.get("ROBOGEN_DIR", "/content/RoboGen"),
                        help="RoboGen の配置ディレクトリ")
    parser.add_argument("--cache-dir", default="/content/drive/MyDrive/robogen_data",
                        help="zip のキャッシュ先（Google Drive 推奨。未マウントなら /content/cache に自動変更）")
    parser.add_argument("--download", action="store_true", help="不足アセットをダウンロードする")
    parser.add_argument("--with-embeddings", action="store_true",
                        help="Objaverse 検索用埋め込み（ステップB 用、約数GB）も取得する")
    parser.add_argument("--verify", action="store_true", help="検証のみ行う")
    args = parser.parse_args()                    # 解析済み引数

    robogen_dir = os.path.abspath(args.robogen_dir)   # RoboGen の絶対パス
    assert os.path.isdir(robogen_dir), f"RoboGen が見つかりません: {robogen_dir}"

    cache_dir = args.cache_dir                    # zip キャッシュ先
    if not os.path.isdir(os.path.dirname(cache_dir)):
        cache_dir = "/content/cache"              # Drive 未マウント時のフォールバック（永続化されない）
        print(f"注意: Drive が未マウントのためキャッシュ先を {cache_dir} にします（セッション終了で消えます）")

    if args.download:
        # --- PartNet-Mobility ---
        ok, _ = verify_partnet(robogen_dir)
        if not ok:
            zip_path = fetch_zip(PARTNET_GDRIVE_ID, os.path.join(cache_dir, "partnet_dataset.zip"))
            extract_to(zip_path, os.path.join(robogen_dir, "data", "dataset"))
        else:
            print("PartNet-Mobility は配置済みです。")
        # --- Objaverse 埋め込み（オプション） ---
        if args.with_embeddings:
            ok, _ = verify_embeddings(robogen_dir)
            if not ok:
                zip_path = fetch_zip(EMBEDDINGS_GDRIVE_ID, os.path.join(cache_dir, "objaverse_embeddings.zip"))
                extract_to(zip_path, os.path.join(robogen_dir, "objaverse_utils", "data"))
            else:
                print("Objaverse 埋め込みは配置済みです。")

    # --- 検証（--download 後も必ず実施して結果を表示する） ---
    all_ok = True                                 # 全体の成否
    for name, fn, required in [("PartNet-Mobility", verify_partnet, True),
                               ("Objaverse 埋め込み", verify_embeddings, False)]:
        ok, msg = fn(robogen_dir)
        mark = "OK " if ok else ("NG " if required else "-- ")   # 必須でないものは NG 扱いにしない
        print(f"[{mark}] {name}: {msg}")
        if required and not ok:
            all_ok = False

    if not all_ok:
        print("\n必須アセットが不足しています。docs/assets_setup.md の手順を確認してください。")
        sys.exit(1)
    print("\nアセット検証に合格しました。")


if __name__ == "__main__":
    main()
