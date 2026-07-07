# ============================================================
# check_env.py
# 役割: Colab 環境のセルフチェック。GPU / import / バージョン / 描画 を検査し、
#       OK/NG の表と診断情報を出力する（ローカルで実行検証できない分、
#       Colab 上での1回の実行から最大限の情報を得るための診断ツール）
# 使い方: python scripts/check_env.py --stage genesis
#         python scripts/check_env.py --stage robogen
# 戻り値: 全チェック合格なら exit 0、1つでも失敗なら exit 1
# ============================================================
import argparse      # コマンドライン引数の解析用
import importlib     # モジュールの動的 import 用
import os            # 環境変数・パス操作用
import sys           # バージョン情報・終了コード用
import traceback     # 失敗時の詳細トレースバック取得用

results = []         # チェック結果の蓄積リスト（(名前, 成否, 詳細) のタプル）


# チェックを1件実行して結果を記録する。fn は詳細文字列を返す関数（例外＝失敗）
def check(name, fn):
    try:
        detail = fn()                            # チェック本体を実行（戻り値は表示用の詳細）
        results.append((name, True, detail or ""))
    except Exception:
        tb = traceback.format_exc(limit=3)       # 直近3フレーム分のトレースバック
        results.append((name, False, tb.strip().splitlines()[-1] + "\n" + tb))


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


# ---------- ステージ定義と実行 ----------

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
}


# チェック結果を表形式で出力し、終了コードを返すメイン処理
def main():
    parser = argparse.ArgumentParser(description="Colab 環境セルフチェック")
    parser.add_argument("--stage", choices=STAGES.keys(), required=True,
                        help="チェック対象ステージ（genesis / robogen）")
    args = parser.parse_args()                   # 解析済みコマンドライン引数

    for name, fn in STAGES[args.stage]:
        check(name, fn)

    # --- 結果表の出力 ---
    print("\n" + "=" * 64)
    print(f" セルフチェック結果 (stage: {args.stage})")
    print("=" * 64)
    all_ok = True                                # 全チェック合格フラグ
    for name, ok, detail in results:
        mark = "OK " if ok else "NG "            # 表示用の合否マーク
        first = detail.splitlines()[0] if detail else ""  # 詳細の先頭行のみ表に表示
        print(f" [{mark}] {name:<24} {first}")
        if not ok:
            all_ok = False

    # --- 失敗があればトレースバック全文を出力（エラー報告に貼ってもらう用） ---
    if not all_ok:
        print("\n----- 失敗の詳細（このログを開発者に共有してください） -----")
        for name, ok, detail in results:
            if not ok:
                print(f"\n### {name}\n{detail}")
        print("\n1つ以上のチェックに失敗しました。")
        sys.exit(1)

    print("\nすべてのチェックに合格しました。次のセルへ進んでください。")
    sys.exit(0)


if __name__ == "__main__":
    main()
