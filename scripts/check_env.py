# ============================================================
# check_env.py
# 役割: Colab 環境のセルフチェック。GPU / import / バージョン / 描画 を検査し、
#       OK/NG の表と診断情報を出力する（ローカルで実行検証できない分、
#       Colab 上での1回の実行から最大限の情報を得るための診断ツール）
# 使い方: python scripts/check_env.py --stage genesis
#         python scripts/check_env.py --stage robogen
#         python scripts/check_env.py --stage recon
# 戻り値: 必須チェック全合格なら exit 0、1つでも失敗なら exit 1
#         （optional なチェックの失敗は [-- ] 表示のみで成否に含めない）
# ============================================================
import argparse      # コマンドライン引数の解析用
import importlib     # モジュールの動的 import 用
import os            # 環境変数・パス操作用
import subprocess    # 外部コマンド（ffmpeg 等）の確認用
import sys           # バージョン情報・終了コード用
import traceback     # 失敗時の詳細トレースバック取得用

results = []         # チェック結果の蓄積リスト（(名前, 成否, 詳細, 必須フラグ) のタプル）


# チェックを1件実行して結果を記録する。fn は詳細文字列を返す関数（例外＝失敗）
def check(name, fn, required=True):
    try:
        detail = fn()                            # チェック本体を実行（戻り値は表示用の詳細）
        results.append((name, True, detail or "", required))
    except Exception:
        tb = traceback.format_exc(limit=3)       # 直近3フレーム分のトレースバック
        results.append((name, False, tb.strip().splitlines()[-1] + "\n" + tb, required))


# モジュールを import してバージョン文字列を返す（import 可否チェックの共通処理）
def try_import(module_name):
    mod = importlib.import_module(module_name)   # 対象モジュール
    ver = getattr(mod, "__version__", "(バージョン属性なし)")  # バージョン表記
    return f"version={ver}"


# ---------- 共通チェック（全ステージで実施） ----------

# Python バージョンが 3.10〜3.13 の範囲内かを検査する
def check_python():
    v = sys.version_info                         # 実行中の Python バージョン
    assert (3, 10) <= (v.major, v.minor) < (3, 14), f"3.10〜3.13 が必要（現在 {v.major}.{v.minor}）"
    return f"{v.major}.{v.minor}.{v.micro}"


# CUDA GPU が利用可能かを torch 経由で検査する
def check_gpu():
    import torch                                 # Colab プリインストールの PyTorch
    assert torch.cuda.is_available(), "CUDA GPU が見つかりません。ランタイム→ランタイムのタイプを変更→GPU を選択してください"
    name = torch.cuda.get_device_name(0)         # GPU デバイス名（T4 等）
    return f"{name} / torch={torch.__version__}"


# ---------- genesis ステージ ----------

# genesis を import して初期化まで通るかを検査する
def check_genesis_import():
    return try_import("genesis")


# genesis を GPU バックエンドで初期化できるかを検査する（失敗時は CPU で再試行）
def check_genesis_init():
    import genesis as gs                         # Genesis 本体
    try:
        gs.init(backend=gs.gpu)                  # GPU (CUDA/Taichi) バックエンドで初期化
        return "backend=gpu"
    except Exception:
        gs.init(backend=gs.cpu)                  # GPU 不可なら CPU フォールバック（遅いが動作可）
        return "backend=cpu（GPU 初期化に失敗したため CPU で動作。性能低下に注意）"


# ---------- robogen ステージ ----------

# PyBullet をヘッドレス (DIRECT) で接続し、1フレームだけオフスクリーン描画できるか検査する
def check_pybullet_headless():
    import pybullet as p                         # PyBullet 物理エンジン
    cid = p.connect(p.DIRECT)                    # GUI なしの DIRECT モードで接続
    img = p.getCameraImage(64, 64)               # 64x64 の1フレームをソフトウェア描画（TinyRenderer）
    p.disconnect(cid)
    assert img is not None and len(img) >= 3, "getCameraImage の戻り値が不正です"
    return "DIRECT 接続 + 1フレーム描画 OK"


# OMPL (モーションプランニング) の import と簡易プランニングを検査する
def check_ompl():
    from ompl import base as ob                  # OMPL の基本モジュール
    from ompl import geometric as og             # 幾何プランニングモジュール
    space = ob.RealVectorStateSpace(2)           # 2次元の状態空間（動作確認用の最小構成）
    space.setBounds(0.0, 1.0)                    # 状態空間の範囲 [0,1]
    ss = og.SimpleSetup(space)                   # プランニング問題のセットアップ
    return "import + 状態空間構築 OK"


# RoboGen 本体が clone・パッチ適用済みかを検査する
def check_robogen_repo():
    robogen_dir = os.environ.get("ROBOGEN_DIR", "/content/RoboGen")  # RoboGen の配置先
    assert os.path.isdir(robogen_dir), f"RoboGen が見つかりません: {robogen_dir}（setup_robogen.sh を先に実行してください）"
    marker = os.path.join(robogen_dir, ".patches_applied")           # パッチ適用済みマーカーファイル
    assert os.path.isfile(marker), "パッチ未適用です（apply_patches.sh を実行してください）"
    with open(marker) as f:
        return f"パッチ適用済み: {f.read().strip()}"


# ---------- recon ステージ（ノートブック③: 動画/画像 → 3DGS + USD） ----------

# GPU の VRAM 総量を確認し、少ない場合は縮小設定を助言する
def check_vram():
    import torch                                 # VRAM 取得用
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3   # VRAM 総量 [GB]
    hint = "" if total_gb >= 14 else "（14GB 未満: GS_CAP_MAX を 500000 に下げることを推奨）"
    return f"VRAM {total_gb:.1f} GB{hint}"


# ffmpeg 本体と mpdecimate フィルタ（重複フレーム除去）の有無を検査する
def check_ffmpeg():
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)   # バージョン確認
    assert out.returncode == 0, "ffmpeg が見つかりません"
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    assert "mpdecimate" in filters.stdout, "mpdecimate フィルタがありません"
    return out.stdout.splitlines()[0][:60]


# 公式 pycolmap（SfM バインディング）の import と CUDA 対応を検査する
def check_pycolmap():
    import pycolmap                              # 公式 COLMAP バインディング
    # SceneManager を持つのは rmbrualla 版（パーサ）。ここに入っていたら隔離失敗＝取り違え
    assert not hasattr(pycolmap, "SceneManager"), \
        "site-packages に rmbrualla/pycolmap が入っています（公式 pycolmap-cuda12 と衝突。setup_recon.sh を確認）"
    has_cuda = getattr(pycolmap, "has_cuda", None)   # CUDA ビルドか（属性名は版差があるため getattr）
    ver = getattr(pycolmap, "__version__", "?")      # バージョン
    return f"version={ver} / has_cuda={has_cuda}"


# gsplat の import・バージョンを検査する
def check_gsplat_import():
    import gsplat                                # 3DGS ラスタライザ
    return f"version={gsplat.__version__}"


# gsplat の CUDA カーネルを JIT コンパイルし、最小レンダリングが通るか検査する
# （初回はコンパイルに 1〜3 分かかる。このチェックがウォームアップを兼ねる）
def check_gsplat_jit():
    import torch                                 # テンソル生成用
    from gsplat import rasterization             # ラスタライズ関数（初回呼び出しで JIT）
    dev = "cuda"                                 # 実行デバイス
    n = 8                                        # テスト用ガウシアン数
    means = torch.randn(n, 3, device=dev) * 0.1 + torch.tensor([0.0, 0.0, 2.0], device=dev)   # 中心座標
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=dev), dim=-1)              # 回転（正規化四元数）
    scales = torch.full((n, 3), 0.05, device=dev)          # スケール
    opacities = torch.full((n,), 0.9, device=dev)          # 不透明度
    colors = torch.rand(n, 3, device=dev)                  # 色（SH なし）
    viewmats = torch.eye(4, device=dev)[None]              # カメラ外部（単位行列）
    Ks = torch.tensor([[[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]]], device=dev)   # カメラ内部
    out, alpha, meta = rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 64, 64)
    assert not torch.isnan(out).any(), "レンダ結果に NaN があります"
    return f"JIT コンパイル + 64x64 レンダ OK（出力 {tuple(out.shape)}）"


# gsplat リポジトリ（examples/simple_trainer.py）と隔離パーサの配置を検査する
def check_gsplat_examples():
    gsplat_dir = os.environ.get("GSPLAT_DIR", "/content/gsplat")               # gsplat clone 先
    trainer = os.path.join(gsplat_dir, "examples", "simple_trainer.py")        # 学習スクリプト
    assert os.path.isfile(trainer), f"{trainer} がありません（setup_recon.sh を実行してください）"
    parser_dir = os.environ.get("GSPLAT_PARSER_DIR", "/content/gsplat_parser") # 隔離パーサの配置先
    parser_pkg = os.path.join(parser_dir, "pycolmap")                          # rmbrualla/pycolmap パッケージ
    assert os.path.isdir(parser_pkg), f"{parser_pkg} がありません（COLMAP パーサの隔離インストール失敗）"
    return "simple_trainer.py + 隔離パーサ OK"


# Open3D の TSDF 融合 API の存在を検査する
def check_open3d_tsdf():
    import open3d as o3d                         # メッシュ抽出用
    assert hasattr(o3d.pipelines.integration, "ScalableTSDFVolume"), "ScalableTSDFVolume がありません"
    return f"version={o3d.__version__} / ScalableTSDFVolume OK"


# OpenCV の ArUco 検出器（スケール校正 aruco 方式）の存在を検査する
def check_aruco():
    import cv2                                   # OpenCV（contrib 込みであること）
    assert hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"), \
        "cv2.aruco.ArucoDetector がありません（opencv-contrib-python が必要）"
    return f"OpenCV {cv2.__version__} / ArucoDetector OK"


# usd-core の import・バージョン・UsdPhysics の存在を検査する
def check_usdcore():
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils   # OpenUSD コアモジュール群
    ver = Usd.GetVersion()                        # OpenUSD バージョン（タプル）
    return f"OpenUSD {'.'.join(map(str, ver))} / UsdPhysics OK"


# 3DGS 標準スキーマ ParticleField3DGaussianSplat の有無を検査する
# （無くても失敗にしない: build_usd.py が .ply sidecar 方式へ自動フォールバックする）
def check_particlefield():
    from pxr import UsdVol                        # ボリュームスキーマ群
    if hasattr(UsdVol, "ParticleField3DGaussianSplat"):
        return "対応（3DGS を USD 標準スキーマで格納します）"
    return "非対応 → .ply sidecar 方式で出力します（Isaac Sim 6.0 でのネイティブ描画には usd-core 26.3+ が必要）"


# transformers に GroundingDINO / SAM2 のクラスがあるか検査する（フェーズ2 用）
def check_transformers_seg():
    import transformers                           # HF transformers
    ok_dino = hasattr(transformers, "GroundingDinoForObjectDetection")   # テキスト→検出
    ok_sam2 = hasattr(transformers, "Sam2Model")                         # マスク生成
    assert ok_dino and ok_sam2, f"GroundingDino={ok_dino}, Sam2={ok_sam2}（transformers>=4.53 が必要）"
    return f"version={transformers.__version__} / GroundingDINO+SAM2 OK"


# ---------- ステージ定義と実行 ----------
# エントリ形式: (名前, 関数) = 必須 / (名前, 関数, False) = optional（失敗しても全体成否に含めない）

STAGES = {
    "genesis": [                                 # ノートブック①用のチェック一覧
        ("Python バージョン", check_python),
        ("GPU (CUDA)", check_gpu),
        ("genesis import", check_genesis_import),
        ("genesis 初期化", check_genesis_init),
        ("imageio import", lambda: try_import("imageio")),
    ],
    "robogen": [                                 # ノートブック②用のチェック一覧
        ("Python バージョン", check_python),
        ("GPU (CUDA)", check_gpu),
        ("pybullet import", lambda: try_import("pybullet")),
        ("pybullet ヘッドレス描画", check_pybullet_headless),
        ("ompl プランニング", check_ompl),
        ("gymnasium import", lambda: try_import("gymnasium")),
        ("stable_baselines3 import", lambda: try_import("stable_baselines3")),
        ("openai SDK import", lambda: try_import("openai")),
        ("RoboGen 配置＋パッチ", check_robogen_repo),
    ],
    "recon": [                                   # ノートブック③用のチェック一覧
        ("Python バージョン", check_python),
        ("GPU (CUDA)", check_gpu),
        ("VRAM 総量", check_vram),
        ("ffmpeg + mpdecimate", check_ffmpeg),
        ("pycolmap (公式/SfM)", check_pycolmap),
        ("gsplat import", check_gsplat_import),
        ("gsplat JIT + 最小レンダ", check_gsplat_jit),
        ("gsplat examples + 隔離パーサ", check_gsplat_examples),
        ("open3d TSDF", check_open3d_tsdf),
        ("trimesh import", lambda: try_import("trimesh")),
        ("plyfile import", lambda: try_import("plyfile")),
        ("OpenCV ArUco", check_aruco),
        ("usd-core + UsdPhysics", check_usdcore),
        ("ParticleField スキーマ", check_particlefield),
        ("genesis import", check_genesis_import),
        ("transformers (DINO/SAM2)", check_transformers_seg, False),
        ("coacd import", lambda: try_import("coacd"), False),
        ("openai SDK import", lambda: try_import("openai"), False),
    ],
}


# チェック結果を表形式で出力し、終了コードを返すメイン処理
def main():
    parser = argparse.ArgumentParser(description="Colab 環境セルフチェック")
    parser.add_argument("--stage", choices=STAGES.keys(), required=True,
                        help="チェック対象ステージ（genesis / robogen / recon）")
    args = parser.parse_args()                   # 解析済みコマンドライン引数

    for entry in STAGES[args.stage]:
        name, fn = entry[0], entry[1]            # チェック名と関数
        required = entry[2] if len(entry) > 2 else True   # 3要素目が無ければ必須扱い（後方互換）
        check(name, fn, required)

    # --- 結果表の出力 ---
    print("\n" + "=" * 64)
    print(f" セルフチェック結果 (stage: {args.stage})")
    print("=" * 64)
    all_ok = True                                # 必須チェックの全合格フラグ
    for name, ok, detail, required in results:
        if ok:
            mark = "OK "                         # 合格
        elif required:
            mark = "NG "                         # 必須の失敗
        else:
            mark = "-- "                         # optional の失敗（成否に含めない）
        first = detail.splitlines()[0] if detail else ""  # 詳細の先頭行のみ表に表示
        print(f" [{mark}] {name:<28} {first}")
        if not ok and required:
            all_ok = False

    # --- 必須の失敗があればトレースバック全文を出力（エラー報告に貼ってもらう用） ---
    if not all_ok:
        print("\n----- 失敗の詳細（このログを開発者に共有してください） -----")
        for name, ok, detail, required in results:
            if not ok and required:
                print(f"\n### {name}\n{detail}")
        print("\n1つ以上の必須チェックに失敗しました。")
        sys.exit(1)

    # optional の失敗は情報として案内する（フェーズ2 実行前に解消すればよい）
    opt_failed = [name for name, ok, _, required in results if not ok and not required]
    if opt_failed:
        print(f"\n注意: オプション項目が未整備です（フェーズ2 で必要）: {', '.join(opt_failed)}")

    print("\nすべての必須チェックに合格しました。次のセルへ進んでください。")
    sys.exit(0)


if __name__ == "__main__":
    main()
