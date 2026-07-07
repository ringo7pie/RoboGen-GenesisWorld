# トラブルシューティング

エラー報告の際は、該当セルの出力全体と `check_env.py` の結果表を添えてください。

## 共通

### 「CUDA GPU が見つかりません」
- メニュー「ランタイム」→「ランタイムのタイプを変更」→ ハードウェアアクセラレータで **T4 GPU** を選択し、
  ランタイム再起動後に最初のセルから実行し直してください。
- 無料枠の GPU 割り当て上限に達している場合は時間をおいて再試行してください。

### セッション切れで環境が消えた
- Colab はセッション終了で `/content` が消えます。ノートブックを最初のセルから再実行してください
  （セットアップは全自動で約10分。Drive にキャッシュがあればアセットの再ダウンロードは不要です）。
- 学習のチェックポイントは `best_model.zip` / `latest_model.zip` として保存されるため、
  重要な結果はセル11 で Drive に保存してください。

### pip の依存解決エラー / Colab 側の更新で壊れた
- Colab のプリインストール（torch / numpy）は時期により更新されます。`check_env.py` の出力に
  実際のバージョンが表示されるので、報告時に添えてください。
- 応急処置として、エラーになったパッケージをバージョン指定なしで `pip install <名前>` し直すと
  解決することがあります。

## ノートブック①（Genesis）

### `gs.init(backend=gs.gpu)` が失敗する
- セルは自動で CPU にフォールバックします（低速だが動作は継続）。
- GPU で使いたい場合はランタイムを再起動して再実行してください（Taichi の CUDA 初期化は
  プロセス再起動で直ることが多い）。

### `show_viewer=True` にしたら GenesisException: No display detected
- Colab にはディスプレイがないためビューアは使えません。`show_viewer=False` に戻してください。

## ノートブック②（RoboGen）

### パッチ適用エラー「RoboGen のコミットが想定と異なる」
- `/content/RoboGen` を削除してセル4を再実行してください: `!rm -rf /content/RoboGen`
- それでも失敗する場合、上流リポジトリの変更が原因の可能性があります。報告してください。

### gdown のダウンロードエラー（Too many users have viewed or downloaded this file）
- 配布リンクへのアクセス集中による一時ブロックです。`docs/assets_setup.md` の
  「手動での代替手順」に従い、ブラウザでダウンロードして自分の Drive に置いてください。

### OMPL の import エラー
- `pip install ompl==2.0.0` が cp310〜cp313 の manylinux wheel を提供しています。
  Colab の Python バージョンが 3.14 以降に更新された場合は wheel がなく失敗します。報告してください。

### ステップA で「mobility.urdf がありません」
- アセット未配置です。セル6（`download_assets.py --download`）を先に実行してください。

### ステップB で OpenAI エラー
- `AuthenticationError`: Colab Secrets の `OPENAI_API_KEY` の値を確認し、
  「ノートブックからのアクセス」が ON になっているか確認してください。
- `model_not_found`: セル8 の `ROBOGEN_LLM_MODEL` を利用可能なモデル（例 `gpt-4o`, `gpt-4.1`）に変更してください。
- レート制限 (`RateLimitError`): 時間をおいて再実行してください。

### 終了時に「free(): invalid size」「corrupted size vs. prev_size」が出る
- OMPL 1.7.0（Boost.Python バインディング）のプロセス終了時のヒープ解放ノイズで、**無害**です。
- `RESULT_MP4=...` の行（結果の mp4 パス）が出力されていれば処理はすべて正常に完了しています。
  成果物（GIF / mp4 / モデル）はエラー表示より前にすべてディスクへ書き込み済みです。

### ステップC が遅い / セッション時間内に終わらない
- 既定の短縮設定（20000 ステップ）は「動作確認」用です。まず短縮設定で最後まで走ることを確認し、
  本格学習は Colab Pro などの長時間ランタイムで `--timesteps 1000000` を指定してください。
- 中断しても `latest_model.zip` が残るため、再開時の初期値として利用できます。

## ノートブック③（動画/画像 → 3DGS + USD）

### gsplat の JIT コンパイルが失敗する（セル5）
- Colab の torch / CUDA 更新が原因の可能性があります。`!nvcc --version` と
  `python -c "import torch; print(torch.__version__, torch.version.cuda)"` の出力を添えて報告してください。
- 応急処置: `pip install gsplat --index-url https://docs.gsplat.studio/whl/pt<torch版>cu<cuda版>` の
  ビルド済み wheel が合えば JIT 不要になります。

### 学習が GPU メモリ不足（OOM）で落ちる（セル10）
- 効果の大きい順: `GS_CAP_MAX = 500000` → セル8 で `LONG_EDGE = 960` にして `--force` 再抽出 → `GS_ITERS = 7000`。

### COLMAP の登録率が低い / SfM が失敗する（セル9）
- 撮影が原因のことがほとんどです（`docs/recon_pipeline.md` の撮影ガイド参照）。
  その場回転だけの動画・無地の壁・ブレが典型例です。
- 画像入力で順序がバラバラな場合は `run_colmap.py --matcher exhaustive` を試してください。

### ParticleField スキーマ非対応と表示される（セル5）
- usd-core が 26.3 未満です。`pip install -U "usd-core>=26.5"` で更新してください。
  非対応のままでも sidecar 方式（gaussians.ply 併置）で出力されるため、進行は可能です。

### Genesis 検証で貫通（CHECK=FAIL）になる（セル16）
- メッシュに穴がある可能性: `configs/recon_defaults.yaml` の `mesh.opacity_min` を 0.1 に、
  `min_cluster_ratio` を 0.05 に上げてセル13 から再実行（`--force`）。
- verify の `z_min` 数値と `static_check.mp4` を添えて報告してください。

### 物体が分離されない / 誤検出（セル19）
- `OBJECT_PROMPTS` は**英語の普通名詞をピリオド区切り**で（例 `"chair. cardboard box."`）。
- しきい値を下げる: `--box-threshold 0.25`。クロップ画像の格子で結果を必ず目視確認してください。

### 質量・摩擦が明らかにおかしい（セル20）
- スケール校正の誤差が3乗で質量に効きます。まず `scale/scale.json` の係数を疑ってください。
- `physics/physics.json` は手修正できます。修正後はセル21 から再実行すれば反映されます。

## 最終フォールバック: condacolab で Python 3.10 環境を作る

Python 3.12 でどうしても解決できない依存問題が出た場合の代替手順です
（condacolab はカーネル再起動を挟むため、セル構成が変わる点に注意）:

```python
# 1つ目のセル（実行後にカーネルが自動再起動する）
!pip install -q condacolab
import condacolab
condacolab.install_miniconda()
```

```python
# 2つ目のセル（再起動後に実行）
!conda create -n robogen python=3.10 -y
# 以降、!conda run -n robogen pip install ... の形で依存を導入し、
# !conda run -n robogen python ... で各スクリプトを実行する
```

ただし conda 環境には torch の再インストール（約 2.5GB）が必要になり、セットアップ時間が大幅に伸びます。
まずは通常手順のエラーを報告することを推奨します。
