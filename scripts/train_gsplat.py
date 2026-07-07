# ============================================================
# train_gsplat.py
# 役割: gsplat の examples/simple_trainer.py をサブプロセスで実行して 3DGS を学習し、
#       標準 .ply（INRIA 形式）を出力する。T4 向けの安全既定（MCMC + ガウシアン数上限）と
#       OOM 検知・具体的な縮小案の提示、ピーク VRAM の記録を行う。
# 重要: gsplat examples は COLMAP パーサとして rmbrualla/pycolmap を要求するが、
#       これは公式 pycolmap（SfM 用）と同名モジュールのため site-packages に同居できない。
#       → setup_recon.sh が隔離ディレクトリに入れた rmbrualla 版を、
#         本スクリプトが「学習サブプロセスの PYTHONPATH 先頭」に注入して解決する。
# 使い方: python train_gsplat.py --workspace <RUN_DIR> [--iters 15000] [--preset indoor|outdoor]
# 出力: <ws>/gsplat/ply/point_cloud_<it-1>.ply ほか（stats/, renders/ 等）
# stdout: RESULT_PLY=<path> / RESULT_NUM_GAUSSIANS=<n> / RESULT_PSNR=<val|NA>
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # 出力 ply の探索用
import json          # 学習統計の読み取り用
import os            # パス・環境変数操作用
import subprocess    # simple_trainer の実行用
import sys           # 終了コード用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import load_recon_config, result_line, save_stats   # 共有ユーティリティ

# OOM 時に表示する縮小案（Colab 1往復で解決できるよう具体値で示す）
OOM_HINTS = """
--- GPU メモリ不足（OOM）の対処（効果の大きい順） ---
1. ガウシアン数上限を下げる: セルの GS_CAP_MAX = 500000
2. フレームを小さくする: 抽出セルで LONG_EDGE = 960 にして --force で再抽出 → COLMAP からやり直し
3. 学習を短くする: GS_ITERS = 7000（品質は下がるがプレビューには十分）
"""


def main():
    parser = argparse.ArgumentParser(description="gsplat simple_trainer による 3DGS 学習")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ（colmap_ws/ がある場所）")
    parser.add_argument("--iters", type=int, default=None, help="学習ステップ数")
    parser.add_argument("--cap-max", type=int, default=None, help="MCMC のガウシアン数上限")
    parser.add_argument("--preset", choices=["indoor", "outdoor"], default="indoor", help="シーンプリセット")
    parser.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc", help="densification 戦略")
    parser.add_argument("--gsplat-dir", default=os.environ.get("GSPLAT_DIR", "/content/gsplat"),
                        help="gsplat リポジトリの場所")
    parser.add_argument("--parser-dir", default=os.environ.get("GSPLAT_PARSER_DIR", "/content/gsplat_parser"),
                        help="rmbrualla/pycolmap の隔離インストール先")
    parser.add_argument("--force", action="store_true", help="既存の学習結果を破棄して再実行")
    args = parser.parse_args()                    # 解析済み引数

    cfg = load_recon_config(args.preset)["gsplat"]   # プリセット適用済みの既定値
    iters = args.iters or cfg["iters"]            # 学習ステップ数
    cap_max = args.cap_max or cfg["cap_max"]      # ガウシアン数上限

    data_dir = os.path.join(args.workspace, "colmap_ws")     # COLMAP 形式データ
    result_dir = os.path.join(args.workspace, "gsplat")      # 学習結果の出力先
    trainer = os.path.join(args.gsplat_dir, "examples", "simple_trainer.py")   # 学習スクリプト
    assert os.path.isfile(os.path.join(data_dir, "sparse", "0", "points3D.bin")), \
        f"COLMAP 結果がありません: {data_dir}/sparse/0（先に run_colmap.py を実行）"
    assert os.path.isfile(trainer), f"{trainer} がありません（setup_recon.sh を実行）"

    existing = sorted(glob.glob(os.path.join(result_dir, "ply", "*.ply")))   # 既存の出力 ply
    if existing and not args.force:
        ply_path = existing[-1]                   # 最新の ply を再利用
        print(f"既存の学習結果を再利用します: {ply_path}（やり直す場合は --force）")
        _report(ply_path, result_dir, args.workspace, reused=True)
        return

    # --- simple_trainer のコマンド組み立て（サブコマンド: mcmc / default） ---
    cmd = [sys.executable, trainer, args.strategy,
           "--data_dir", data_dir,
           "--result_dir", result_dir,
           "--data_factor", "1",                  # 抽出時に縮小済みのため 1
           "--max_steps", str(iters),
           "--save_ply",                          # 標準 .ply を出力する
           "--ply_steps", str(iters),             # 最終ステップで ply 保存
           "--save_steps", str(iters),            # ckpt も最終のみ（ディスク節約）
           "--eval_steps", str(iters),            # 評価も最終のみ（PSNR 取得用）
           "--disable_viewer",                    # Colab ではビューア不可
           "--disable_video",                     # 軌道レンダ動画は生成しない（時間節約）
           ]
    if args.strategy == "mcmc":
        cmd += ["--strategy.cap-max", str(cap_max)]   # ガウシアン数上限（T4 VRAM 対策）

    # --- 学習サブプロセスの環境: 隔離した rmbrualla/pycolmap を PYTHONPATH 先頭へ注入 ---
    env = os.environ.copy()                       # 親環境のコピー
    env["PYTHONPATH"] = args.parser_dir + os.pathsep + env.get("PYTHONPATH", "")

    print("実行コマンド:", " ".join(cmd))
    print(f"（strategy={args.strategy}, iters={iters}, cap_max={cap_max}, preset={args.preset}）")

    # --- 実行（ログはそのまま流しつつ末尾を保持して OOM 判定に使う） ---
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(trainer), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []                                     # ログ末尾のバッファ（OOM 判定用）
    for line in proc.stdout:
        print(line, end="")
        tail.append(line)
        if len(tail) > 200:
            tail.pop(0)
    proc.wait()

    if proc.returncode != 0:
        log_tail = "".join(tail)                  # 末尾ログ
        if "out of memory" in log_tail.lower() or "cuda oom" in log_tail.lower():
            print("\nエラー: GPU メモリ不足で学習が停止しました。" + OOM_HINTS)
        else:
            print(f"\nエラー: 学習が異常終了しました（exit {proc.returncode}）。上のログ全体を報告してください。")
        sys.exit(1)

    plys = sorted(glob.glob(os.path.join(result_dir, "ply", "*.ply")))   # 生成された ply
    assert plys, f"学習は終了しましたが .ply が見つかりません: {result_dir}/ply"
    _report(plys[-1], result_dir, args.workspace, reused=False,
            extra={"iters": iters, "cap_max": cap_max, "strategy": args.strategy, "preset": args.preset})


# 出力 .ply の検証・統計記録・RESULT_ 行の出力を行う
# 引数: ply_path = 出力 ply、result_dir = 学習結果ディレクトリ、workspace = 作業ディレクトリ
def _report(ply_path, result_dir, workspace, reused, extra=None):
    from plyfile import PlyData                   # ply ヘッダ検査用
    n_gauss = PlyData.read(ply_path)["vertex"].count   # ガウシアン数
    assert n_gauss > 1000, f"ガウシアン数が異常に少ないです（{n_gauss}）。学習ログを確認してください。"

    psnr = "NA"                                   # 最終評価の PSNR（取得できた場合のみ）
    peak_vram_gb = "NA"                           # ピーク VRAM（stats から）
    for stat_file in sorted(glob.glob(os.path.join(result_dir, "stats", "val_step*.json"))):
        try:
            with open(stat_file) as f:
                data = json.load(f)               # 評価統計
            psnr = round(float(data.get("psnr", data.get("PSNR", "nan"))), 2)
        except Exception:
            pass
    try:
        import torch                              # ピーク VRAM の取得（同一プロセスではないため参考値）
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.mem_get_info()[1] / 1024**3 -
                                 torch.cuda.mem_get_info()[0] / 1024**3, 1)   # 使用中 VRAM [GB]
    except Exception:
        pass

    stats = {"ply": ply_path, "num_gaussians": int(n_gauss), "psnr": psnr,
             "used_vram_gb_now": peak_vram_gb, "reused": reused}
    stats.update(extra or {})
    save_stats(workspace, "gsplat", stats)
    print(f"3DGS 学習完了: {ply_path}（ガウシアン数 {n_gauss}, PSNR {psnr}）")
    result_line("PLY", ply_path)
    result_line("NUM_GAUSSIANS", n_gauss)
    result_line("PSNR", psnr)


if __name__ == "__main__":
    main()
