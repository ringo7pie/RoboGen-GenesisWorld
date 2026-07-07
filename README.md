# RoboGen + Genesis World on Google Colab

[RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen)（LLM によるロボットタスク自動生成 + スキル学習）と
[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)（生成 AI 時代の物理シミュレータ）を
**Google Colab の無料 GPU で動かす**ためのノートブックとセットアップ一式です。

> **注意**: RoboGen の公式 Genesis 統合版は未公開です（公開版 RoboGen は PyBullet 実装）。
> 本リポジトリでは ① Genesis 単体デモ、② RoboGen（PyBullet 版）の Colab 移植、の2つを段階的に提供します。

## 1. 概要

| ノートブック | 内容 | 所要時間 | 必要なもの |
|---|---|---|---|
| [notebooks/01_genesis_demo.ipynb](notebooks/01_genesis_demo.ipynb) | Genesis を Colab でヘッドレス実行し、Franka アームのシミュレーション動画を生成 | ~10分 | GPU ランタイムのみ |
| [notebooks/02_robogen_colab.ipynb](notebooks/02_robogen_colab.ipynb) | RoboGen のフルパイプライン: タスク再生 → LLM タスク生成 → スキル学習 | ~1時間 | GPU + OpenAI API キー |
| [notebooks/03_recon_to_usd.ipynb](notebooks/03_recon_to_usd.ipynb) | **Real2Sim**: 動画/複数画像 → 3D Gaussian Splatting（.ply）＋ 物理パラメータ付き USD（Isaac Sim 互換）→ Genesis 物理検証 | ~2〜3時間 | GPU + 撮影した動画（フェーズ2 のみ OpenAI キー） |

## 2. 必要なもの

- Google アカウント（Colab の GPU ランタイム。無料枠の T4 で動作します）
- OpenAI API キー（ノートブック②のステップB のみ。再生・学習だけならキー不要）

## 3. クイックスタート（ノートブック①）

1. このリポジトリを GitHub で開き、`notebooks/01_genesis_demo.ipynb` → 「Open in Colab」
   （または Colab の「GitHub」タブから本リポジトリの URL を指定して開く）
2. メニュー「ランタイム」→「ランタイムのタイプを変更」→ **T4 GPU** を選択
3. 上からセルを順に実行 → 最後に Franka アームのシミュレーション動画 (mp4) が表示されれば成功

## 4. ノートブック②の事前準備

### 4.1 OpenAI キーの登録（ステップB を使う場合）

1. Colab 画面左側の**カギアイコン（シークレット）**をクリック
2. 「新しいシークレットを追加」→ 名前 `OPENAI_API_KEY`、値に API キーを貼り付け
3. 「ノートブックからのアクセス」を **ON**

### 4.2 アセットについて

PartNet-Mobility（パース済み版）はノートブックが自動ダウンロードします（**ライセンス登録不要**）。
Google Drive をマウントするとダウンロードした zip がキャッシュされ、次回以降のセッションで再利用されます。
詳細と手動での代替手順は [docs/assets_setup.md](docs/assets_setup.md) を参照してください。

## 5. ノートブック②の実行手順

| ステップ | 内容 | 期待される出力 | 所要時間 |
|---|---|---|---|
| セットアップ（セル1〜6） | RoboGen clone・依存導入・パッチ適用・アセット取得 | セルフチェック全 OK | ~15分 |
| **A. 再生**（セル7） | 例題タスク「ノート PC を開く」のシーン構築 | 周回カメラの mp4 | ~10分 |
| **B. タスク生成**（セル8〜9） | 説明文から LLM がタスク config・報酬関数を生成 | 生成された YAML の表示 | ~10分 + API 課金 |
| **C. スキル学習**（セル10〜11） | モーションプランニング + RL (SB3 SAC) | 学習過程・結果の mp4、Drive 保存 | 短縮設定で ~30分 |

**ステップA が通れば環境構築は成功**です。まず A まで確実に動かしてから B / C に進んでください。

## 6. リポジトリ構成

```
├── notebooks/
│   ├── 01_genesis_demo.ipynb     # ① Genesis 単体デモ
│   ├── 02_robogen_colab.ipynb    # ② RoboGen フルパイプライン
│   └── 03_recon_to_usd.ipynb     # ③ Real2Sim（動画→3DGS→物理付き USD）
├── scripts/
│   ├── setup_genesis.sh          # ① 用セットアップ
│   ├── setup_robogen.sh          # ② 用セットアップ一括（clone・依存・パッチ）
│   ├── setup_recon.sh            # ③ 用セットアップ（gsplat・COLMAP パーサ隔離）
│   ├── apply_patches.sh          # パッチ適用（冪等）
│   ├── check_env.py              # 環境セルフチェック（--stage genesis|robogen|recon）
│   ├── genesis_record.py         # Genesis カメラ録画ユーティリティ（①②③共用）
│   ├── download_assets.py        # ② アセット取得・検証
│   ├── run_robogen_task.py       # ② タスク実行 CLI（--mode replay|learn）
│   ├── recon_utils.py            # ③ 共有（COLMAP パーサ/3DGS ply/TSDF/座標変換）
│   ├── extract_frames.py         # ③ フレーム抽出（fps・重複/ブレ除去・縮小）
│   ├── run_colmap.py             # ③ SfM（pycolmap GPU・登録率診断）
│   ├── train_gsplat.py           # ③ 3DGS 学習（gsplat・OOM 診断）
│   ├── extract_mesh.py           # ③ TSDF 衝突メッシュ + 床分離 + Z-up 変換
│   ├── calibrate_scale.py        # ③ 実寸校正（manual/aruco/gpt の3方式）
│   ├── segment_objects.py        # ③ 物体分離（GroundingDINO+SAM2+視点投票）
│   ├── estimate_physics.py       # ③ GPT-4o 物性推定（materials_db で clamp）
│   ├── build_usd.py              # ③ USD 構築（UsdPhysics + ParticleField 3DGS + usdz）
│   └── verify_genesis_usd.py     # ③ Genesis 物理検証（落下テスト + mp4）
├── patches/                      # RoboGen への修正パッチ（詳細は patches/README.md）
├── robogen_overrides/            # RoboGen へコピーする置き換えモジュール
│   ├── gym/__init__.py           # gym→gymnasium 互換 shim
│   └── RL/sb3_learn.py           # RL 学習部の stable-baselines3 実装
├── requirements/                 # 依存定義（genesis.txt / robogen.txt / recon.txt）
├── configs/                      # colab_paths / recon_defaults / materials_db
└── docs/                         # アセット手順・撮影/Real2Sim ガイド・トラブルシューティング
```

## 6.5 ノートブック③（Real2Sim）の実行手順

| ステップ | 内容 | 期待される出力 |
|---|---|---|
| セットアップ（セル1〜6） | gsplat/pycolmap/usd-core 等の導入 + セルフチェック | 全 OK（ParticleField 対応可否も表示） |
| 入力〜**関門1**（セル7〜11） | 動画→フレーム→COLMAP→3DGS 学習 | `.ply`（SuperSplat で閲覧可） |
| フェーズ1〜**関門2**（セル13〜16) | 衝突メッシュ→スケール校正→静的 USD→Genesis 球落下検証 | `scene_static.usdc` + 検証 mp4 |
| フェーズ2〜**関門3**（セル18〜22） | SAM2 物体分離→GPT-4o 物性→動的 USD→落下静止検証 | `scene_dynamic.usdc` + 検証 mp4 |
| 仕上げ（セル23） | usdz 化 + Drive 保存 | `scene.usdz` ほか一式 |

撮影ガイド・Isaac Sim への持ち込み手順は [docs/recon_pipeline.md](docs/recon_pipeline.md) を参照。

## 7. 技術的な設計判断

- **Python 3.12 ネイティブ**: RoboGen 公式の conda 環境 (Python 3.9) は Colab で再現できないため、
  Colab 標準の Python 3.12 で動くよう依存を置換（condacolab は不採用。理由は壊れやすさとカーネル再起動）
- **RL: ray 1.13 → stable-baselines3**: `ray.rllib.agents` は Python 3.12 で動作不能のため、
  `run_RL()` を同一インターフェースで SB3 SAC/PPO に再実装（`robogen_overrides/RL/sb3_learn.py`）
- **gym → gymnasium shim**: 旧 gym 0.21 はインストール不可。`import gym` を gymnasium に橋渡しする
  互換パッケージを RoboGen 直下に配置（LLM 生成済みタスクファイルの `import gym` も無修正で動く）
- **openai 新 SDK**: 旧 0.27 呼び出しをパッチで >=1.x に移行。モデル名は `ROBOGEN_LLM_MODEL` で変更可能
- **OMPL**: ソースビルド不要。公式 PyPI wheel（`pip install ompl`）を使用
- **RoboGen はコミット固定 + .patch 方式**: fork を持たず、`git apply` できる差分として修正を管理
  （一覧と理由は [patches/README.md](patches/README.md)）
- **torch / numpy は Colab プリインストールを流用**: 再インストールによる CUDA 不整合を回避
- **［③］3DGS は gsplat（Apache-2.0）+ 公式 pycolmap**: INRIA 実装・MASt3R 系は非商用ライセンスのため不採用。
  gsplat examples が要求する rmbrualla/pycolmap（公式版と同名衝突）は隔離ディレクトリ＋学習サブプロセスのみの PYTHONPATH 注入で共存
- **［③］3DGS の USD 格納は OpenUSD 26.03 標準スキーマ `ParticleField3DGaussianSplat`**（Isaac Sim 6.0 ネイティブ描画）。
  非対応環境には .ply sidecar + customData で自動フォールバック
- **［③］物理は「見た目=3DGS / 衝突=TSDF メッシュ」の二層構成**（NVIDIA NuRec と同型）。
  物性は GPT-4o 推定値を materials_db.yaml の物理的レンジで clamp し、質量=密度×実寸体積で決定

## 8. トラブルシューティング

[docs/troubleshooting.md](docs/troubleshooting.md) を参照してください。
報告の際は `check_env.py` の結果表とエラーが出たセルの出力全体を添えてください。

## 9. ライセンス・謝辞

- [RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen)（MIT License）— Wang et al., *RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation* (ICML 2024)
- [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)（Apache-2.0 License）
- [PartNet-Mobility / SAPIEN](https://sapien.ucsd.edu/) データセット
- [Objaverse](https://objaverse.allenai.org/) データセット

本リポジトリのコード（scripts / patches / notebooks）は MIT License です。
