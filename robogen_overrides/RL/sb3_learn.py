# ============================================================
# sb3_learn.py
# 役割: RoboGen の RL 学習部 (RL/ray_learn.py, ray 1.13 RLlib ベース) を
#       stable-baselines3 (SB3) で置き換えたモジュール。
#       ray 1.13 は Python 3.12 / Colab で動作しないため、run_RL() の
#       インターフェース（引数・戻り値）を維持したまま SB3 SAC/PPO で再実装する。
# 使い方: execute.py が `from RL.sb3_learn import run_RL` として呼ぶ
#        （パッチ 02 で import を差し替え）。
# 環境変数:
#   ROBOGEN_RL_TIMESTEPS     : 総学習ステップ数の上書き（Colab 短縮デモ用。デフォルトは呼び出し側の値）
#   ROBOGEN_RL_EVAL_INTERVAL : 評価間隔ステップ数（デフォルト 20000）
# 配置: setup_robogen.sh がこのファイルを RoboGen/RL/ へコピーする
# ============================================================
import os                        # 環境変数・パス操作用
import pickle                    # 最良軌跡の状態保存用
import numpy as np               # 観測・報酬の数値処理用
import gymnasium                 # SB3 が要求する環境 API（gym shim 経由と同一オブジェクト）

from manipulation.utils import save_env, save_numpy_as_gif   # RoboGen 側の状態保存・GIF 保存


# RoboGen の旧 gym API 環境（reset()→obs、step()→4値）を
# gymnasium API（reset()→(obs,info)、step()→5値）へ変換するラッパー
# 引数: env = RoboGen の SimpleEnv（旧 API）
class OldToNewAPIWrapper(gymnasium.Env):
    def __init__(self, env):
        self.env = env                                        # ラップ対象の RoboGen 環境
        self.observation_space = env.observation_space        # 観測空間（gym shim 経由なので gymnasium の空間オブジェクト）
        self.action_space = env.action_space                  # 行動空間
        self.metadata = getattr(env, "metadata", {"render_modes": []})   # SB3 が参照するメタデータ

    # gymnasium 形式の reset: (観測, info辞書) を返す
    def reset(self, *, seed=None, options=None):
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)                               # 旧 API の seed() に橋渡し
        obs = self.env.reset()                                # 旧 API は観測のみ返す
        return np.asarray(obs, dtype=np.float32), {}

    # gymnasium 形式の step: (観測, 報酬, terminated, truncated, info) の5値を返す
    def step(self, action):
        obs, reward, done, info = self.env.step(action)       # 旧 API は4値
        # 旧 API の done を terminated として扱う（時間切れ判定は info に無いため truncated=False）
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), False, info

    # 描画は RoboGen 側の実装（rgb, depth を返す）へ委譲する
    def render(self):
        return self.env.render()

    # 終了処理: PyBullet 接続を切断する
    def close(self):
        if hasattr(self.env, "disconnect"):
            self.env.disconnect()


# タスク config から RoboGen 環境を構築する（ray_learn.make_env と同一インターフェース）
# 引数: config = env_config 辞書、render = GUI 表示フラグ（Colab では常に False）
# 戻り値: RoboGen の SimpleEnv（旧 gym API のまま）
def make_env(config, render=False):
    from manipulation.utils import build_up_env              # 循環 import 回避のため関数内 import（元実装踏襲）

    env, safe_config = build_up_env(
        config["task_config_path"],                           # タスク定義 YAML のパス
        config["solution_path"],                              # サブステップ等の置き場
        config["task_name"],                                  # 学習対象サブステップ名
        config["last_restore_state_file"],                    # 直前サブステップの最終状態ファイル
        render=render,
        action_space=config["action_space"],                  # 行動空間の種類（delta-translation 等）
        randomize=config["randomize"],                        # シーンのランダム化フラグ
        obj_id=config["obj_id"],                              # 候補オブジェクトの選択インデックス
    )
    return env


# RL でサブステップのスキルを学習する（ray_learn.run_RL と同一の引数・戻り値）
# 戻り値: (最良モデルの保存パス, 最良ロールアウトの RGB フレーム列, 最良軌跡の状態ファイルパス列)
def run_RL(task_config_path, solution_path, task_name, last_restore_state_file, save_path,
           action_space="delta-translation", algo="sac", timesteps_total=1000000, load_policy_path=None, seed=0,
           render=False, randomize=False, use_bard=True, obj_id=0,
           use_gpt_size=True, use_gpt_joint_angle=True, use_gpt_spatial_relationship=True,
           use_distractor=False):
    from stable_baselines3 import SAC, PPO                    # SB3 のアルゴリズム（重いので関数内 import）

    # --- Colab の時間制約に合わせた学習量の上書き（環境変数が最優先） ---
    timesteps_total = int(os.environ.get("ROBOGEN_RL_TIMESTEPS", timesteps_total))   # 総学習ステップ数
    eval_interval = int(os.environ.get("ROBOGEN_RL_EVAL_INTERVAL", 20000))           # 評価間隔

    env_config = {                                            # make_env に渡す環境設定（元実装と同じキー構成）
        "task_config_path": task_config_path,
        "solution_path": solution_path,
        "task_name": task_name,
        "last_restore_state_file": last_restore_state_file,
        "action_space": action_space,
        "randomize": randomize,
        "use_bard": use_bard,
        "obj_id": obj_id,
        "use_gpt_size": use_gpt_size,
        "use_gpt_joint_angle": use_gpt_joint_angle,
        "use_gpt_spatial_relationship": use_gpt_spatial_relationship,
        "use_distractor": use_distractor,
    }

    raw_env = make_env(env_config, render=render)             # RoboGen 環境（旧 API・評価ロールアウトにも使用）
    env = OldToNewAPIWrapper(raw_env)                         # SB3 用の gymnasium ラッパー

    # --- アルゴリズムの構築（元実装の SAC/PPO 設定に相当する規模のネットワーク） ---
    if algo == "sac":
        model = SAC("MlpPolicy", env, seed=seed, verbose=0,
                    learning_starts=1000,                     # 元実装 (ray sac) と同じウォームアップ
                    policy_kwargs={"net_arch": [256, 256, 256]})   # 元実装の Q/policy 3層 256 に合わせる
    elif algo == "ppo":
        model = PPO("MlpPolicy", env, seed=seed, verbose=0,
                    n_steps=2048, batch_size=128,
                    policy_kwargs={"net_arch": [128, 128]})   # 元実装の fcnet_hiddens [128,128] に合わせる
    else:
        raise ValueError(f"未対応のアルゴリズムです: {algo}（sac / ppo のみ）")

    if load_policy_path is not None and os.path.exists(str(load_policy_path)):
        model = model.load(load_policy_path, env=env)         # 学習済みポリシーからの再開

    best_state_save_path = os.path.join(save_path, "best_state")   # 最良軌跡の状態保存先
    os.makedirs(best_state_save_path, exist_ok=True)

    timesteps = 0                                             # これまでの総学習ステップ数
    eval_time = 1                                             # 評価の実施回数
    best_ret = -np.inf                                        # これまでの最良リターン
    best_rgbs = None                                          # 最良ロールアウトの RGB フレーム列
    best_state_files = None                                   # 最良軌跡の状態ファイルパス列
    best_model_path = os.path.join(save_path, "best_model.zip")    # 最良モデルの保存パス

    # --- 学習ループ: eval_interval ごとに評価ロールアウトを行い最良を残す（元実装と同じ構造） ---
    while timesteps < timesteps_total:
        chunk = min(eval_interval, timesteps_total - timesteps)    # 今回学習するステップ数
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        timesteps += chunk
        print(f"学習進捗: {timesteps}/{timesteps_total} ステップ")

        model.save(os.path.join(save_path, "latest_model"))    # 最新モデルの保存（セッション切れ対策）

        # --- 評価ロールアウト（決定的方策で1エピソード実行し、状態と映像を記録） ---
        obs = raw_env.reset()                                  # 旧 API: 観測のみ返る
        done = False                                           # エピソード終了フラグ
        ret = 0.0                                              # 累積リターン
        rgbs = []                                              # RGB フレーム列
        state_files = []                                       # 状態ファイルパス列
        states = []                                            # 状態オブジェクト列
        t_idx = 0                                              # ステップインデックス
        state_save_path = os.path.join(save_path, f"eval_{eval_time}")   # 今回の評価の保存先
        os.makedirs(state_save_path, exist_ok=True)
        while not done:
            action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)   # 決定的行動
            obs, reward, done, info = raw_env.step(action)     # 旧 API の4値
            ret += reward
            rgb, depth = raw_env.render()                      # オフスクリーン描画（rgb, depth）
            rgbs.append(rgb)

            state_file_path = os.path.join(state_save_path, f"state_{t_idx}.pkl")   # 状態ファイルの保存先
            state = save_env(raw_env, save_path=state_file_path)                    # 環境状態の保存
            state_files.append(state_file_path)
            states.append(state)
            t_idx += 1

        save_numpy_as_gif(np.array(rgbs), os.path.join(state_save_path, "execute.gif"))
        print(f"評価 {eval_time} 回目: {timesteps} ステップ時点のリターン = {ret:.3f}")
        eval_time += 1

        # --- 最良の更新（元実装と同じく、状態・GIF・モデルを best_state/ に残す） ---
        if ret > best_ret:
            best_ret = ret
            model.save(best_model_path.replace(".zip", ""))    # SB3 は拡張子 .zip を自動付与
            best_rgbs = rgbs
            best_state_files = state_files
            for idx, state in enumerate(states):
                with open(os.path.join(best_state_save_path, f"state_{idx}.pkl"), "wb") as f:
                    pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(best_state_save_path, f"return_{round(ret, 3)}.txt"), "w") as f:
                f.write(str(ret))
            save_numpy_as_gif(np.array(best_rgbs), os.path.join(best_state_save_path, "best.gif"))

    raw_env.disconnect()                                       # PyBullet 接続の切断
    return best_model_path, best_rgbs, best_state_files
