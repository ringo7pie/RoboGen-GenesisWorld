# パッチ一覧

適用先: [Genesis-Embodied-AI/RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen)
コミット `486612cd0baa40857b561419c5a42fbb7e67724f`（2024-05-30）に固定。
適用は `scripts/apply_patches.sh`（`git apply`、冪等）で行う。

| パッチ | 対象ファイル | 目的 |
|---|---|---|
| `01_openai_new_sdk.patch` | `gpt_4/query.py` | 旧 openai SDK (0.27) → 新 SDK (>=1.x) へ移行。API キーはコード埋め込みをやめ環境変数 `OPENAI_API_KEY` を参照。モデル名は環境変数 `ROBOGEN_LLM_MODEL` で上書き可能（gpt-4 廃止対策） |
| `02_sb3_rl.patch` | `execute.py` | RL 学習部の import を `RL.ray_learn`（ray 1.13、Python 3.12 非対応）から `RL.sb3_learn`（stable-baselines3 実装、`robogen_overrides/RL/sb3_learn.py` として同梱）へ差し替え |
| `03_skip_lavis_verify.patch` | `objaverse_utils/find_uid_utils.py` | BLIP-2 (salesforce-lavis、Python 3.12 非対応) による物体画像検証を既定でスキップ（環境変数 `ROBOGEN_SKIP_VERIFY=0` で従来動作）。lavis の import を遅延化 |

## パッチ以外の置き換え（robogen_overrides/）

| ファイル | コピー先 | 目的 |
|---|---|---|
| `robogen_overrides/gym/__init__.py` | `RoboGen/gym/` | `import gym` を gymnasium へ橋渡しする shim（旧 gym 0.21 は Python 3.12 にインストール不可） |
| `robogen_overrides/RL/sb3_learn.py` | `RoboGen/RL/` | `run_RL()` の SB3 再実装（ray_learn.py と同一インターフェース） |

## 運用ルール

- パッチが 10 本を超えて肥大化した場合は、fork リポジトリ方式への移行を検討する。
- RoboGen のコミットを更新する場合は、全パッチを `git apply --check` で再検証してから
  `setup_robogen.sh` の `ROBOGEN_COMMIT` を更新すること。
