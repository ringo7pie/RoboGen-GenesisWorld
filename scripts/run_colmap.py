# ============================================================
# run_colmap.py
# 役割: 公式 pycolmap（GPU SIFT）でフレーム集合から SfM を実行し、
#       gsplat の Parser が読める配置（<ws>/colmap_ws/images + sparse/0）を作る
# 処理: 特徴抽出 → マッチング（sequential/exhaustive）→ インクリメンタル SfM →
#       最大モデルを sparse/0 に配置 → 登録率で3段階診断（OK/警告/中止）
# 使い方: python run_colmap.py --workspace <RUN_DIR> [--matcher sequential|exhaustive] [--no-gpu]
# 出力: <ws>/colmap_ws/{images/, database.db, sparse/0/}, colmap_stats.json
# stdout: RESULT_SPARSE=<dir> / RESULT_REGISTERED=<n>/<total>
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # フレーム列挙用
import os            # パス操作用
import shutil        # ファイル配置用
import sys           # 終了コード用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import load_recon_config, result_line, save_stats   # 共有ユーティリティ

# 撮影ガイド（登録率が低いときに表示する対処法）
CAPTURE_HINTS = """
--- SfM がうまくいかないときの撮影のコツ（docs/recon_pipeline.md 参照） ---
1. カメラを「ゆっくり・横に動かす」（その場回転だけの映像は視差がなく復元できません）
2. 特徴の少ない無地の壁・床だけを写さない（家具・模様など目印を画角に入れる）
3. 明るい環境で撮る（ブレ・ノイズを減らす）
4. 対策オプション: --matcher exhaustive（時間はかかるが頑健）/ 抽出 fps を上げる（--fps 4）
"""


def main():
    parser = argparse.ArgumentParser(description="pycolmap による SfM 実行")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ（frames/ がある場所）")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default=None,
                        help="マッチング方式（動画=sequential / 無順序画像=exhaustive）")
    parser.add_argument("--no-gpu", action="store_true", help="GPU SIFT を使わない（フォールバック）")
    parser.add_argument("--force", action="store_true", help="既存の SfM 結果を破棄して再実行")
    args = parser.parse_args()                    # 解析済み引数

    import pycolmap                               # 公式 COLMAP バインディング（重いので main 内 import）

    cfg = load_recon_config()["colmap"]           # 既定値
    matcher = args.matcher or cfg["matcher"]      # マッチング方式

    frames_dir = os.path.join(args.workspace, "frames")          # 入力フレーム
    ws = os.path.join(args.workspace, "colmap_ws")               # COLMAP 作業ディレクトリ
    images_dir = os.path.join(ws, "images")                      # gsplat Parser が読む画像配置
    db_path = os.path.join(ws, "database.db")                    # 特徴量データベース
    sparse_dir = os.path.join(ws, "sparse")                      # SfM 出力
    sparse0 = os.path.join(sparse_dir, "0")                      # 採用モデルの配置先

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))   # フレーム一覧
    assert frames, f"フレームがありません: {frames_dir}（先に extract_frames.py を実行）"

    if os.path.isfile(os.path.join(sparse0, "points3D.bin")) and not args.force:
        print("既存の SfM 結果を再利用します（やり直す場合は --force）")
        result_line("SPARSE", sparse0)
        return
    if os.path.isdir(ws):
        shutil.rmtree(ws)                         # --force またはゴミがある場合は作り直す
    os.makedirs(images_dir, exist_ok=True)

    # --- 画像を colmap_ws/images へ配置（symlink。gsplat Parser と COLMAP の両方が参照） ---
    for p in frames:
        os.symlink(os.path.abspath(p), os.path.join(images_dir, os.path.basename(p)))

    device = pycolmap.Device.cpu if args.no_gpu else pycolmap.Device.auto   # SIFT の実行デバイス

    # --- 特徴抽出（動画フレームは同一カメラ想定 → SINGLE で内部パラメータを共有） ---
    print(f"[1/3] 特徴抽出（device={'cpu' if args.no_gpu else 'auto(GPU)'}）...")
    pycolmap.extract_features(db_path, images_dir,
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              device=device)

    # --- マッチング ---
    print(f"[2/3] マッチング（{matcher}）...")
    if matcher == "sequential":
        pycolmap.match_sequential(db_path)        # 連番前提の近傍マッチング（動画向け・高速）
    else:
        pycolmap.match_exhaustive(db_path)        # 全対マッチング（無順序画像向け・低速）

    # --- インクリメンタル SfM ---
    print("[3/3] インクリメンタル SfM ...")
    os.makedirs(sparse_dir, exist_ok=True)
    recs = pycolmap.incremental_mapping(db_path, images_dir, sparse_dir)   # {index: Reconstruction}
    assert recs, ("SfM が1つもモデルを構築できませんでした。" + CAPTURE_HINTS)

    # --- 最大モデル（登録画像が最も多いもの）を採用し、sparse/0 に統一配置 ---
    best_idx = max(recs.keys(), key=lambda k: recs[k].num_reg_images())    # 採用モデルの index
    best = recs[best_idx]                          # 採用モデル
    if best_idx != 0:
        shutil.rmtree(sparse0, ignore_errors=True)
        os.makedirs(sparse0, exist_ok=True)
        best.write(sparse0)                        # 0 番へ書き直し
        print(f"モデル {best_idx} を sparse/0 として採用")

    # --- 品質診断 ---
    n_total = len(frames)                          # 入力フレーム数
    n_reg = best.num_reg_images()                  # 登録できた画像数
    ratio = n_reg / n_total                        # 登録率
    mean_err = best.compute_mean_reprojection_error()          # 平均再投影誤差 [px]
    n_points = len(best.points3D)                  # 疎点群の点数
    print(f"登録率: {n_reg}/{n_total} ({ratio:.0%}) / 平均再投影誤差: {mean_err:.2f}px / 疎点数: {n_points}")

    save_stats(args.workspace, "colmap", {
        "registered": n_reg, "total": n_total, "register_ratio": ratio,
        "mean_reprojection_error_px": mean_err, "num_points3d": n_points,
        "matcher": matcher, "gpu": not args.no_gpu, "num_models": len(recs)})

    if ratio < cfg["min_register_ratio"]:
        print(f"エラー: 登録率が {ratio:.0%} と低すぎます（下限 {cfg['min_register_ratio']:.0%}）。" + CAPTURE_HINTS)
        sys.exit(1)
    if ratio < cfg["warn_register_ratio"]:
        print(f"警告: 登録率が {ratio:.0%} と低めです。品質が不足する場合は撮影し直しを検討してください。" + CAPTURE_HINTS)

    result_line("SPARSE", sparse0)
    result_line("REGISTERED", f"{n_reg}/{n_total}")


if __name__ == "__main__":
    main()
