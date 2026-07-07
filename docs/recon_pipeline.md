# Real2Sim パイプライン（ノートブック③）ガイド

動画/複数画像 → 3D Gaussian Splatting ＋ 物理パラメータ付き USD を生成する
`notebooks/03_recon_to_usd.ipynb` の撮影ガイド・仕組み・調整方法をまとめます。

## 1. 撮影ガイド（入力データの作り方）

SfM（COLMAP）と 3DGS の品質は撮影でほぼ決まります。

### 基本
- **静止したシーン**を、**カメラを動かしながら**撮る（動く人・ペットが写り込まない時間帯に）
- スマホの通常ビデオで可（1080p 推奨、1〜3分）。写真なら 50〜300 枚
- **ゆっくり・横に移動**しながら対象を写す。**その場で回転するだけの映像は復元できません**（視差が出ないため）
- 同じ場所を別の高さ・角度からもう1周撮ると品質が上がる

### 避けること
- 速いパン・手ブレ（ブレたフレームは自動除去されるが、多すぎると枚数不足になる）
- 無地の壁・床だけの画角（特徴点が取れない）。家具や模様を常に画角に入れる
- 逆光・暗所・鏡やガラスの大写し

### スケール校正の準備（どれか1つ）
| 方式 | 準備 | 精度 |
|---|---|---|
| manual | シーン内の1つの物の寸法を実測しておく（例: 机の高さ 72cm） | 高 |
| aruco | [ArUco マーカー](https://chev.me/arucogen/)（DICT_4X4_50）を印刷し、**一辺の実寸を測って**床や机に置いて撮影 | 高 |
| gpt | 不要（ドア・ペットボトル等の既知物体が写っていれば GPT-4o が推定） | 低（±20〜30%） |

## 2. パイプラインの仕組み

```
入力（動画/画像）
 → extract_frames.py   フレーム抽出（fps・重複除去 mpdecimate・ブレ除去・縮小）
 → run_colmap.py       SfM: カメラポーズ+疎点群（pycolmap GPU）
 → train_gsplat.py     3DGS 学習（gsplat v1.5.3 / MCMC 戦略・ガウシアン数上限）
 → extract_mesh.py     深度レンダ→TSDF 融合→衝突メッシュ、床平面分離、Z-up 変換
 → calibrate_scale.py  実寸換算係数（manual/aruco/gpt）
 → build_usd.py        UsdPhysics 準拠 USD（+3DGS を ParticleField 格納）
 → verify_genesis_usd.py  Genesis で球落下/物体落下の物理検証 + mp4
 →（フェーズ2）segment_objects.py → estimate_physics.py → 動的剛体入り USD
```

### 座標とスケールの扱い
- COLMAP の出力はスケール不定・向き任意。メッシュとガウシアンは COLMAP 単位のまま持ち、
  `mesh/transform.json`（床基準 Z-up 化）と `scale/scale.json`（m 換算）を **USD 出力時に焼き込み**ます
- そのため**スケール校正のやり直しはセル14 → セル15 以降の再実行だけ**で済みます（メッシュ再抽出は不要）

### 3DGS の USD 格納
- usd-core 26.3+ では **OpenUSD 標準スキーマ `UsdVol.ParticleField3DGaussianSplat`** に格納します
  （Isaac Sim 6.0 / Omniverse RTX がネイティブ描画）
- スキーマ非対応環境では `gaussians.ply` を sidecar 出力し、USD の customData から参照します
- どちらになったかは `usd/report.json` の `gs_mode` を参照
- 注意: SH（視点依存色）係数は座標回転に対して未回転のまま格納します（見た目はわずかな近似）

### 物理の構造（UsdPhysics）
- 静的: 床=`UsdGeom.Plane`+CollisionAPI、シーンメッシュ=CollisionAPI+三角メッシュ衝突（approximation "none"）
- 動的（フェーズ2）: RigidBodyAPI + MassAPI（質量=GPT推定密度×実寸体積）+ coacd 凸分解
  （approximation "convexHull"）+ PhysicsMaterialAPI（静/動摩擦・反発）
- 単位系: Z-up / metersPerUnit=1 / kilogramsPerUnit=1 / 重力 9.81 m/s²（-Z）

## 3. パラメータ調整の勘所

| 症状 | 調整 |
|---|---|
| GPU メモリ不足 | セル10 `GS_CAP_MAX=500000`、セル8 `LONG_EDGE=960` |
| 3DGS がぼやける | `GS_ITERS=30000`（+30〜50分）、撮影のやり直し |
| メッシュに穴・浮遊ゴミ | `configs/recon_defaults.yaml` の `mesh.min_cluster_ratio` を上げる（0.05）、`opacity_min` を上げる（0.1） |
| 屋外で遠景がメッシュを汚す | セル10/13 の `SCENE_PRESET="outdoor"` |
| 床の検出が変 | 床が画角に十分写っているか確認。`mesh.floor_dist_ratio` を調整 |
| 物体が検出されない | セル19 のプロンプトを写っている物の英単語に変更、`--box-threshold 0.25` |
| 物性値がおかしい | セル20 の表を確認し `physics.json` を手修正 → セル21 から再実行 |

## 4. Isaac Sim / Omniverse への持ち込み

1. Drive の `recon_outputs/<RUN_ID>/final/` から `scene_dynamic.usdc`（または `scene.usdz`）を取得
2. **Isaac Sim 6.0 以降**で File > Open。3DGS（ParticleField）は RTX レンダラでそのまま描画されます
   - Isaac Sim 5.x / 他の DCC では 3DGS プリムは描画されません（メッシュと物理は機能します）
3. Play で物理開始。動的物体は質量・摩擦付きの剛体として挙動します
4. うまく動かない場合は `usd/report.json` の `validation` と `warnings` を確認

### sidecar 方式（gs_mode=sidecar）だった場合
- `scene_*.usdc` と `gaussians.ply` を**同じフォルダに置いて**配布してください
- 3DGS の表示には SuperSplat（https://superspl.at/editor）等の外部ビューアを使用します

## 5. 出力ファイル一覧

| パス（Drive: recon_outputs/<RUN_ID>/final/） | 内容 |
|---|---|
| `scene_static.usdc` | 静的シーン USD（床+衝突メッシュ+3DGS+重力） |
| `scene_dynamic.usdc` | 動的剛体入り USD（フェーズ2 実施時） |
| `scene.usdz` | 単一ファイルパッケージ |
| `scene_*.usda` | デバッグ用テキスト版 |
| `point_cloud_*.ply` | 3DGS（INRIA 形式） |
| `gaussians.ply` | 焼き込み済み sidecar（sidecar 方式時のみ） |
| `static_check.mp4` / `dynamic_check.mp4` | Genesis 物理検証動画 |
| `report.json` | 格納方式・Compliance 結果・警告 |

中間生成物（フレーム・COLMAP・メッシュ・スケール等）は `recon_outputs/<RUN_ID>/backup/` に
バックアップされ、セル6 の `RESUME_RUN_ID` で別セッションから再開できます。
