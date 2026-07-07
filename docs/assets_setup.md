# アセットのセットアップ手順

RoboGen の実行には以下のアセットが必要です。**基本はすべてノートブック②のセルが自動取得します**が、
仕組みと手動での代替手順をここにまとめます。

## 1. PartNet-Mobility（必須）

関節付き 3D オブジェクト（引き出し・扉・家電など）のデータセットです。タスクのシーン構築に必須です。

### 自動取得（推奨）

RoboGen 公式がパース済み版を Google Drive で配布しており、ノートブック②の
アセット取得セル（`scripts/download_assets.py --download`）が `gdown` で自動ダウンロードします。
**SAPIEN サイトへのライセンス登録は不要です。**

- 取得元: RoboGen README 記載の公開リンク（ファイル ID `1d-1txzcg_ke17NkHKAolXlfDnmPePFc6`）
- 配置先: `RoboGen/data/dataset/<オブジェクトID>/mobility.urdf`
- キャッシュ: ダウンロードした zip は Google Drive の `MyDrive/robogen_data/` に保存され、
  次回セッションでは再ダウンロードせずに済みます（Drive 未マウント時はキャッシュされません）

### 手動での代替手順（自動取得が失敗する場合）

Google Drive の公開リンクは一定期間のダウンロード集中で一時的にブロックされることがあります
（gdown が "Too many users have viewed or downloaded this file recently" エラーを出す）。その場合:

1. ブラウザで https://drive.google.com/file/d/1d-1txzcg_ke17NkHKAolXlfDnmPePFc6/view を開き、手動でダウンロード
2. zip を自分の Google Drive の `MyDrive/robogen_data/partnet_dataset.zip` にアップロード
3. ノートブック②のアセット取得セルを再実行（キャッシュとして認識され、展開だけが行われます）

それでも入手できない場合は、[SAPIEN 公式サイト](https://sapien.ucsd.edu/browse) でアカウント登録して
オリジナル版をダウンロードできますが、RoboGen は URDF から関節ツリーを抽出した**パース済み版が前提**のため、
公式配布版の利用を強く推奨します。

## 2. Objaverse アセット（自動・操作不要）

タスクに登場する小物（マグカップ・ノート等）の 3D モデルです。
実行時に `objaverse` パッケージが**必要な UID だけをオンデマンドでダウンロード**します。
ライセンス登録は不要で、ユーザー操作もありません。

- 保存先: `RoboGen/objaverse_utils/data/obj/<uid>/`
- 例題タスク（`example_tasks/`）の YAML には検証済み UID が記載されているため、
  そのままダウンロード・利用されます

## 3. Objaverse 検索用の文埋め込み（ステップB でのみ必要）

LLM で**新規タスクを生成**するとき、タスクに必要な新しい物体を Objaverse 全体（80万件）から
文埋め込みの類似度で検索します。この事前計算済み埋め込み（RoboGen 公式配布）が必要です。

- 取得: ノートブック②のステップB 準備セル（`download_assets.py --download --with-embeddings`）が自動取得
  （ファイル ID `1dFDpG3tlckTUSy7VYdfkNqtfVctpn3T6`、数 GB）
- 配置先: `RoboGen/objaverse_utils/data/*.pt`
- ステップA（再生）だけを試す場合は不要です

## 容量の目安

| アセット | サイズ目安 | 置き場所 |
|---|---|---|
| PartNet-Mobility パース済み版 (zip) | 数 GB | Drive キャッシュ + Colab ローカル展開 |
| Objaverse 埋め込み (zip) | 数 GB | 同上 |
| Objaverse 個別オブジェクト | 数 MB〜数十 MB / 個 | Colab ローカル |

Google Drive の無料枠は 15GB のため、キャッシュを置く場合は空き容量に注意してください。
Drive を使わない場合もノートブックは動作します（毎セッション再ダウンロードになるだけです）。
