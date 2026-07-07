# ============================================================
# run_robogen_task.py
# 役割: RoboGen のタスク実行をラップする CLI（ノートブック②から呼ばれる）
#   --mode replay : タスク config からシーンを構築し、周回カメラ映像を mp4 化する
#                   （学習なし・OpenAI キー不要。環境構築成功の関門）
#   --mode learn  : サブステップごとにモーションプランニング＋RL(SB3) でスキルを学習する
# 前提: setup_robogen.sh 実行済み（RoboGen 配置＋パッチ適用済み）
# 使用例:
#   python run_robogen_task.py --mode replay --task-config example_tasks/Open_Laptop/....yaml
#   python run_robogen_task.py --mode learn  --task-config ... --timesteps 50000
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # 生成された GIF の探索用
import os            # パス・環境変数操作用
import sys           # モジュール検索パス操作用
import time
import datetime


# GIF を mp4 に変換する（ノートブックでのインライン再生用。GIF は Colab 上で重いため）
# 引数: gif_path = 変換元 GIF、mp4_path = 出力先 mp4
# 戻り値: 出力した mp4 のパス
def gif_to_mp4(gif_path, mp4_path):
    import imageio                                # 動画入出力（遅延 import）
    reader = imageio.get_reader(gif_path)         # GIF リーダー
    fps = reader.get_meta_data().get("duration")  # 1フレームの表示時間 (ms)
    fps = 1000.0 / fps if fps else 10             # duration から fps へ換算（不明なら 10fps）
    writer = imageio.get_writer(mp4_path, fps=fps, macro_block_size=1)   # mp4 ライター
    for frame in reader:
        writer.append_data(frame)                 # フレームを順次書き込み
    writer.close()
    print(f"mp4 変換完了: {mp4_path}")
    return mp4_path


# メイン処理: 引数を解析し、RoboGen の execute() を適切なモードで呼び出す
def main():
    parser = argparse.ArgumentParser(description="RoboGen タスク実行ラッパー")
    parser.add_argument("--mode", choices=["replay", "learn"], required=True,
                        help="replay=シーン再構築＋動画化（キー不要） / learn=スキル学習")
    parser.add_argument("--task-config", required=True,
                        help="タスク定義 YAML のパス（RoboGen ディレクトリからの相対パス）")
    parser.add_argument("--robogen-dir", default=os.environ.get("ROBOGEN_DIR", "/content/RoboGen"),
                        help="RoboGen の配置ディレクトリ")
    parser.add_argument("--output", default=None,
                        help="結果 mp4 のコピー先パス（省略時は変換のみ）")
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="[learn] 1サブステップあたりの RL 総学習ステップ数（Colab 短縮デモ既定 20000。本格学習は 1000000）")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="[learn] 評価ロールアウトの間隔ステップ数")
    parser.add_argument("--algo", default="RL_sac", choices=["RL_sac", "RL_ppo"],
                        help="[learn] 学習アルゴリズム")
    parser.add_argument("--obj-id", type=int, default=0,
                        help="候補オブジェクトの選択インデックス")
    args = parser.parse_args()                    # 解析済み引数

    # --- RoboGen は相対パス前提のため、作業ディレクトリと import パスを揃える ---
    robogen_dir = os.path.abspath(args.robogen_dir)   # RoboGen の絶対パス
    assert os.path.isdir(robogen_dir), f"RoboGen が見つかりません: {robogen_dir}（setup_robogen.sh を先に実行）"
    os.chdir(robogen_dir)
    sys.path.insert(0, robogen_dir)

    assert os.path.isfile(args.task_config), f"タスク config が見つかりません: {args.task_config}"

    # --- 学習量の上書き（sb3_learn.py が環境変数を参照する） ---
    os.environ["ROBOGEN_RL_TIMESTEPS"] = str(args.timesteps)
    os.environ["ROBOGEN_RL_EVAL_INTERVAL"] = str(args.eval_interval)

    import execute as ex                          # RoboGen の実行モジュール（パッチ適用済み）

    time_string = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")   # 実行 ID（保存先の識別に使用）

    # --- タスク config から solution_path（成果物の置き場）を読む ---
    import yaml                                   # タスク config の解析用
    with open(args.task_config) as f:
        task_config = yaml.safe_load(f)
    solution_path = None                          # サブステップ・成果物の置き場
    for obj in task_config:
        if "solution_path" in obj:
            solution_path = obj["solution_path"]
            break
    assert solution_path, "タスク config に solution_path がありません"

    if args.mode == "replay":
        # test_env() はモジュールグローバル task_config_path を参照するため（上流実装の仕様）、
        # 呼び出し前にモジュール属性として設定しておく
        ex.task_config_path = args.task_config
        try:
            ex.execute(args.task_config, time_string=time_string, run_training=False,
                       gui=False, obj_id=args.obj_id)
        except SystemExit:
            pass                                  # test_env 後の exit() は正常終了として扱う
        # 生成された周回カメラ GIF（construction.gif）を mp4 化する
        gif_path = os.path.join(solution_path, "blip2", time_string, "construction.gif")   # test_env の出力先
        assert os.path.isfile(gif_path), f"シーン構築 GIF が生成されていません: {gif_path}"
        mp4_path = gif_path.replace(".gif", ".mp4")   # 変換後の mp4 パス
        gif_to_mp4(gif_path, mp4_path)

    else:  # learn
        ex.execute(args.task_config, time_string=time_string, run_training=True,
                   training_algo=args.algo, gui=False, obj_id=args.obj_id,
                   use_motion_planning=True)
        # 全サブステップ連結 GIF（all-<time_string>.gif）を mp4 化する
        gif_path = os.path.join(solution_path, f"all-{time_string}.gif")   # execute() の最終出力
        if not os.path.isfile(gif_path):
            # 連結 GIF が無い場合は最後のサブステップの execute.gif を代わりに使う
            candidates = sorted(glob.glob(os.path.join(solution_path, "**", time_string, "**", "execute.gif"),
                                          recursive=True))   # 個別サブステップの GIF 一覧
            assert candidates, f"学習結果の GIF が見つかりません: {solution_path}"
            gif_path = candidates[-1]
        mp4_path = gif_path.replace(".gif", ".mp4")
        gif_to_mp4(gif_path, mp4_path)

    # --- 指定があれば結果 mp4 をコピー（Drive 保存等はノートブック側で行う） ---
    if args.output:
        import shutil                             # ファイルコピー用
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        shutil.copy(mp4_path, args.output)
        print(f"結果をコピーしました: {args.output}")

    print(f"RESULT_MP4={mp4_path}")               # ノートブックがこの行をパースして表示に使う


if __name__ == "__main__":
    main()
