# ============================================================
# extract_frames.py
# 役割: 動画 / 画像フォルダ / 画像 zip から、SfM（COLMAP）に適したフレーム集合を作る
#   - 動画: ffmpeg で fps 抽出 + mpdecimate（重複除去）+ 長辺縮小
#   - 画像: EXIF 回転を正規化して長辺縮小コピー
#   - 共通: Laplacian 分散の下位（ブレ画像）を除去し、max_frames まで等間隔間引き
# 使い方: python extract_frames.py --input <動画|フォルダ|zip> --workspace <RUN_DIR>
# 出力: <workspace>/frames/frame_%05d.jpg, frames_meta.json
# stdout: RESULT_FRAMES=<dir> / RESULT_FRAME_COUNT=<n>
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # 画像ファイルの列挙用
import os            # パス操作用
import shutil        # ファイルコピー・削除用
import subprocess    # ffmpeg 実行用
import sys           # 終了コード用
import zipfile       # zip 入力の展開用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import load_recon_config, result_line, save_stats   # 共有ユーティリティ

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}   # 画像として扱う拡張子
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}     # 動画として扱う拡張子


# 動画から ffmpeg でフレームを抽出する（fps 指定 + 重複除去 + 長辺縮小）
# 戻り値: 抽出したフレーム数
def extract_from_video(video_path, out_dir, fps, long_edge):
    os.makedirs(out_dir, exist_ok=True)
    # scale の if() 内のカンマは ffmpeg フィルタ構文上エスケープが必要
    vf = (f"fps={fps},mpdecimate,"
          f"scale=if(gt(iw\\,ih)\\,{long_edge}\\,-2):if(gt(iw\\,ih)\\,-2\\,{long_edge})")   # フィルタ列
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path,
           "-vf", vf, "-fps_mode", "vfr", "-qscale:v", "2",
           os.path.join(out_dir, "frame_%05d.jpg")]                  # 抽出コマンド
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"ffmpeg が失敗しました:\n{res.stderr[-2000:]}"
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))


# 画像フォルダから EXIF 回転を正規化しつつ縮小コピーする
# 戻り値: コピーした枚数
def copy_from_images(src_dir, out_dir, long_edge):
    from PIL import Image, ImageOps               # EXIF 正規化・リサイズ用
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(p for p in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True)
                   if os.path.splitext(p)[1].lower() in IMAGE_EXTS)   # 入力画像一覧
    for i, path in enumerate(files):
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # EXIF 回転を実画素に反映
        w, h = img.size                           # 元サイズ
        scale = long_edge / max(w, h)             # 長辺基準の縮小率
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(os.path.join(out_dir, f"frame_{i + 1:05d}.jpg"), quality=92)
    return len(files)


# Laplacian 分散（ピントの鋭さ指標）が下位のフレームを削除する
# 戻り値: (残存数, 削除数)
def drop_blurry(frame_dir, drop_ratio):
    import cv2                                    # ブレ判定用
    files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))   # フレーム一覧
    if drop_ratio <= 0 or len(files) < 10:
        return len(files), 0
    scores = []                                   # (鋭さスコア, パス)
    for p in files:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)                # グレースケール読込
        scores.append((cv2.Laplacian(gray, cv2.CV_64F).var(), p)) # Laplacian 分散
    scores.sort(key=lambda t: t[0])
    n_drop = int(len(files) * drop_ratio)         # 削除枚数
    for _, p in scores[:n_drop]:
        os.remove(p)
    return len(files) - n_drop, n_drop


# フレーム数が max を超えていたら等間隔に間引く
# 戻り値: 残存数
def thin_to_max(frame_dir, max_frames):
    files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))   # フレーム一覧
    if len(files) <= max_frames:
        return len(files)
    keep_idx = set(round(i * (len(files) - 1) / (max_frames - 1)) for i in range(max_frames))   # 残すインデックス
    for i, p in enumerate(files):
        if i not in keep_idx:
            os.remove(p)
    return len(glob.glob(os.path.join(frame_dir, "*.jpg")))


# ファイル名を frame_%05d.jpg へ連番リネームし直す（間引き後の欠番を詰める）
def renumber(frame_dir):
    files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))   # フレーム一覧
    for i, p in enumerate(files):
        tmp = os.path.join(frame_dir, f"__tmp_{i:05d}.jpg")       # 一時名（衝突回避）
        os.rename(p, tmp)
    for i, p in enumerate(sorted(glob.glob(os.path.join(frame_dir, "__tmp_*.jpg")))):
        os.rename(p, os.path.join(frame_dir, f"frame_{i + 1:05d}.jpg"))


def main():
    parser = argparse.ArgumentParser(description="SfM 用フレーム抽出")
    parser.add_argument("--input", required=True, help="動画ファイル / 画像フォルダ / 画像 zip")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ（出力は <ws>/frames）")
    parser.add_argument("--fps", type=float, default=None, help="動画からの抽出レート")
    parser.add_argument("--long-edge", type=int, default=None, help="長辺の縮小サイズ [px]")
    parser.add_argument("--max-frames", type=int, default=None, help="最大フレーム数")
    parser.add_argument("--min-frames", type=int, default=None, help="最小フレーム数（下回ると中止）")
    parser.add_argument("--blur-drop-ratio", type=float, default=None, help="ブレ除去の割合")
    parser.add_argument("--force", action="store_true", help="既存出力を破棄して再実行")
    args = parser.parse_args()                    # 解析済み引数

    cfg = load_recon_config()["frames"]           # 既定値
    fps = args.fps if args.fps is not None else cfg["fps"]
    long_edge = args.long_edge or cfg["long_edge"]
    max_frames = args.max_frames or cfg["max_frames"]
    min_frames = args.min_frames or cfg["min_frames"]
    blur_ratio = args.blur_drop_ratio if args.blur_drop_ratio is not None else cfg["blur_drop_ratio"]

    out_dir = os.path.join(args.workspace, "frames")   # 出力先
    if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "*.jpg")) and not args.force:
        n = len(glob.glob(os.path.join(out_dir, "*.jpg")))
        print(f"既存のフレーム {n} 枚を再利用します（やり直す場合は --force）")
        result_line("FRAMES", out_dir); result_line("FRAME_COUNT", n)
        return
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)                    # --force 時は作り直す

    # --- 入力種別の判定と展開 ---
    src = args.input                              # 入力パス
    ext = os.path.splitext(src)[1].lower()        # 拡張子
    if ext == ".zip":
        unzip_dir = os.path.join(args.workspace, "input_unzipped")   # zip 展開先
        if os.path.isdir(unzip_dir):
            shutil.rmtree(unzip_dir)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(unzip_dir)
        src, ext = unzip_dir, ""                  # 以降はフォルダとして処理

    if os.path.isdir(src):
        n_raw = copy_from_images(src, out_dir, long_edge)            # 画像入力
        input_kind = "images"                     # 入力種別（stats 用）
    elif ext in VIDEO_EXTS:
        n_raw = extract_from_video(src, out_dir, fps, long_edge)     # 動画入力
        input_kind = "video"
    else:
        print(f"エラー: 入力の種別を判定できません: {args.input}（動画/フォルダ/zip を指定）")
        sys.exit(1)
    print(f"抽出/コピー: {n_raw} 枚")

    # --- ブレ除去 → 間引き → 連番整理 ---
    n_after_blur, n_dropped = drop_blurry(out_dir, blur_ratio)
    print(f"ブレ除去: {n_dropped} 枚削除（Laplacian 分散の下位 {blur_ratio:.0%}）")
    n_final = thin_to_max(out_dir, max_frames)
    renumber(out_dir)
    print(f"最終フレーム数: {n_final}")

    assert n_final >= min_frames, (
        f"フレームが {n_final} 枚しかありません（最低 {min_frames} 枚）。"
        f"動画を長くする・fps を上げる（--fps 4）・ブレの少ない映像にする、を試してください。")

    save_stats(args.workspace, "frames", {
        "input": args.input, "input_kind": input_kind, "raw_count": n_raw,
        "blur_dropped": n_dropped, "final_count": n_final,
        "fps": fps, "long_edge": long_edge})
    result_line("FRAMES", out_dir)
    result_line("FRAME_COUNT", n_final)


if __name__ == "__main__":
    main()
