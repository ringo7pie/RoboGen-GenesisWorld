# ============================================================
# calibrate_scale.py
# 役割: COLMAP 復元（スケール不定）を実寸[m]に合わせる換算係数を求める。3方式を切替可能:
#   manual : 既知寸法の手入力。(a) 3面図から読み取った COLMAP 距離 + 実測値、または
#            (b) フレーム上の2点 (u,v) 指定 → 最近傍の COLMAP 疎点間距離で換算
#   aruco  : 実寸既知の ArUco マーカーを全フレームから検出し、
#            COLMAP ポーズで4隅を多視点三角測量 → 辺長中央値から換算
#   gpt    : GPT-4o に「寸法が標準化された既知物体」を選ばせ、その bbox 内の疎点の
#            Z-up 高さ/幅と典型実寸から換算（confidence: low）
# 使い方例:
#   python calibrate_scale.py --workspace <ws> --mode manual --real-dist 0.72 --colmap-dist 1.85
#   python calibrate_scale.py --workspace <ws> --mode manual --real-dist 0.72 \
#          --points "frame=frame_00050.jpg;a=512,300;b=512,880"
#   python calibrate_scale.py --workspace <ws> --mode aruco --marker-size 0.15
#   python calibrate_scale.py --workspace <ws> --mode gpt
# 出力: <ws>/scale/scale.json / stdout: RESULT_SCALE=<factor>
# ============================================================
import argparse      # コマンドライン引数の解析用
import base64        # gpt 方式の画像送信用
import glob          # フレーム列挙用
import json          # 応答解析用
import os            # パス操作用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import (load_colmap_model, load_transform, apply_transform,
                         result_line, save_scale, save_stats)


# ---------- manual 方式 ----------

# "frame=NAME;a=u,v;b=u,v" 形式の指定を解析する
def parse_points_arg(spec):
    parts = dict(kv.split("=", 1) for kv in spec.split(";"))      # key=value に分解
    pa = tuple(float(x) for x in parts["a"].split(","))           # 点 a の (u,v)
    pb = tuple(float(x) for x in parts["b"].split(","))           # 点 b の (u,v)
    return parts["frame"], pa, pb


# 指定フレーム上の 2D 点に最も近い COLMAP 疎点（3D）を返す
# 戻り値: (3D 座標, 2D 距離[px])
def nearest_sparse_point(model, frame_name, uv):
    for im in model["images"].values():
        if im["name"] != frame_name:
            continue
        valid = im["point3D_ids"] >= 0            # 3D 点に対応づいたキーポイントのみ
        xys = im["xys"][valid]                    # 2D 位置
        ids = im["point3D_ids"][valid]            # 対応する 3D 点 ID
        assert len(xys) > 0, f"{frame_name} に三角測量済みの特徴点がありません"
        d2 = np.linalg.norm(xys - np.array(uv), axis=1)   # 2D 距離
        k = int(d2.argmin())                      # 最近傍のインデックス
        # points3D は id→index 辞書を作っていないため線形探索を避けて再読込せず、ID 一致で取得する
        idx = model["_id2idx"][ids[k]]            # 3D 点配列のインデックス
        return model["points_xyz"][idx], float(d2[k])
    raise AssertionError(f"フレームが見つかりません: {frame_name}")


def mode_manual(args, model):
    assert args.real_dist and args.real_dist > 0, "--real-dist（実測距離 [m]）を指定してください"
    if args.colmap_dist:                          # (a) 3面図読み取り方式
        colmap_dist = float(args.colmap_dist)     # COLMAP 単位の距離
        details = {"input": "colmap_dist", "colmap_dist": colmap_dist}
    else:                                         # (b) 2点指定方式
        assert args.points, "--colmap-dist か --points のどちらかを指定してください"
        frame, pa, pb = parse_points_arg(args.points)
        p3a, err_a = nearest_sparse_point(model, frame, pa)   # 点 a の最近傍 3D 点
        p3b, err_b = nearest_sparse_point(model, frame, pb)   # 点 b の最近傍 3D 点
        colmap_dist = float(np.linalg.norm(p3a - p3b))        # 3D 距離（COLMAP 単位）
        if max(err_a, err_b) > 20:
            print(f"警告: 指定点と特徴点の距離が {max(err_a, err_b):.0f}px 離れています（精度低下の可能性）")
        details = {"input": "points", "frame": frame, "px_err": [err_a, err_b],
                   "colmap_dist": colmap_dist}
    scale = args.real_dist / colmap_dist          # 換算係数 [m/COLMAP単位]
    return scale, "high", details


# ---------- aruco 方式 ----------

def mode_aruco(args, model, frames_dir):
    import cv2                                    # ArUco 検出用
    assert args.marker_size and args.marker_size > 0, "--marker-size（マーカー一辺 [m]）を指定してください"
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))   # マーカー辞書
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())        # 検出器

    name2view = {v["name"]: v for v in model["views"]}   # フレーム名→ビュー
    detections = {}                                # {marker_id: [(view, corners[4,2]) ...]}
    n_tried = 0                                    # 検出を試したフレーム数
    for path in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
        name = os.path.basename(path)              # フレーム名
        if name not in name2view:
            continue                                # SfM に登録されなかったフレームは使えない
        n_tried += 1
        corners, ids, _ = detector.detectMarkers(cv2.imread(path))   # マーカー検出
        if ids is None:
            continue
        for c, mid in zip(corners, ids.flatten()):
            detections.setdefault(int(mid), []).append((name2view[name], c.reshape(4, 2)))

    usable = {mid: obs for mid, obs in detections.items() if len(obs) >= 2}   # 2視点以上で見えたマーカー
    assert usable, (f"ArUco マーカーを2視点以上で検出できませんでした（検出試行 {n_tried} フレーム）。"
                    "マーカーが小さすぎる/ブレている可能性があります。manual 方式への切替を検討してください。")

    side_lengths = []                              # 三角測量した辺長（COLMAP 単位）
    for mid, obs in usable.items():
        corners3d = []                             # 4隅の 3D 位置
        for ci in range(4):
            pts2d = [o[1][ci] for o in obs]        # 各視点での隅の 2D 位置
            views = [o[0] for o in obs]            # 対応ビュー
            corners3d.append(triangulate_dlt(views, pts2d))
        corners3d = np.stack(corners3d)            # [4,3]
        for i in range(4):
            side_lengths.append(float(np.linalg.norm(corners3d[i] - corners3d[(i + 1) % 4])))
    side_lengths = np.array(side_lengths)
    median_side = float(np.median(side_lengths))   # 辺長の中央値
    cv_ratio = float(side_lengths.std() / (median_side + 1e-12))   # 変動係数
    if cv_ratio > 0.10:
        print(f"警告: マーカー辺長のばらつきが大きいです（CV={cv_ratio:.0%}）。検出品質を確認してください。")
    scale = args.marker_size / median_side         # 換算係数
    return scale, "high", {"markers": {str(k): len(v) for k, v in usable.items()},
                           "median_side_colmap": median_side, "cv": cv_ratio,
                           "marker_size_m": args.marker_size}


# 複数視点の 2D 観測から DLT で 3D 点を三角測量する
def triangulate_dlt(views, pts2d):
    rows = []                                      # DLT の係数行列
    for view, (u, v) in zip(views, pts2d):
        P = view["K"] @ view["w2c"][:3, :]         # 射影行列 [3,4]
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)                             # [2M,4]
    _, _, vt = np.linalg.svd(A)                    # 最小二乗解
    X = vt[-1]
    return X[:3] / X[3]


# ---------- gpt 方式 ----------

GPT_PROMPT = """あなたはシーン画像から実寸の手がかりを見つける専門家です。
画像に写っている「寸法が世界的に標準化・ほぼ一定の物体」を最大3つ選び、JSON で返してください。
例: 室内ドア(高さ≈2.0m)、A4用紙(0.297m)、500mlペットボトル(高さ≈0.21m)、
    椅子の座面高(≈0.45m)、レターサイズの標識、ギターなど。
各物体について: name(物体名), typical_dimension_m(その典型寸法[メートル]),
dimension_type("height"=鉛直方向 か "width"=水平方向),
bbox([x0,y0,x1,y1] 画像を0-1で正規化した外接矩形), confidence(0-1) を返すこと。
確実な物体が無ければ objects を空配列にすること。"""

GPT_SCHEMA = {   # structured output 用の JSON スキーマ
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"},
                       "typical_dimension_m": {"type": "number"},
                       "dimension_type": {"type": "string", "enum": ["height", "width"]},
                       "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                       "confidence": {"type": "number"}},
        "required": ["name", "typical_dimension_m", "dimension_type", "bbox", "confidence"],
        "additionalProperties": False}}},
    "required": ["objects"], "additionalProperties": False}


def mode_gpt(args, model, frames_dir, workspace):
    from openai import OpenAI                     # GPT-4o 呼び出し用
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY が未設定です（セル18 を先に実行）"
    client = OpenAI()                             # API クライアント
    llm_model = os.environ.get("RECON_LLM_MODEL", "gpt-4o")   # 使用モデル

    T = load_transform(workspace)                 # COLMAP→Z-up 変換（高さ測定に使用）
    name2im = {im["name"]: im for im in model["images"].values()}   # フレーム名→画像情報

    reg_names = sorted(name2im.keys())            # SfM 登録済みフレーム
    picks = [reg_names[int(i * (len(reg_names) - 1) / 3)] for i in range(4)]   # 代表4枚を等間隔で選ぶ
    estimates = []                                 # 各推定 (scale, name)
    details_objs = []                              # 詳細記録
    for name in dict.fromkeys(picks):              # 重複除去して順序維持
        path = os.path.join(frames_dir, name)      # フレームのパス
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()   # 画像を base64 化
        res = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": GPT_PROMPT},
                      {"role": "user", "content": [
                          {"type": "text", "text": "この画像から寸法既知の物体を選んでください。"},
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "known_objects", "strict": True, "schema": GPT_SCHEMA}},
            temperature=0)
        objs = json.loads(res.choices[0].message.content)["objects"]   # 推定物体リスト
        im = name2im[name]                          # このフレームの特徴点情報
        valid = im["point3D_ids"] >= 0              # 3D 対応のあるキーポイント
        xys = im["xys"][valid]                      # 2D 位置
        idxs = np.array([model["_id2idx"][pid] for pid in im["point3D_ids"][valid]])   # 3D 点 index
        W, H = _frame_size(path)                    # フレームの実サイズ
        for ob in objs:
            if ob["confidence"] < 0.5:
                continue                             # 低信頼はスキップ
            x0, y0, x1, y1 = ob["bbox"]              # 正規化 bbox
            in_box = ((xys[:, 0] >= x0 * W) & (xys[:, 0] <= x1 * W) &
                      (xys[:, 1] >= y0 * H) & (xys[:, 1] <= y1 * H))   # bbox 内の点
            if in_box.sum() < 10:
                continue                             # 点が少なすぎる
            pts_zup = apply_transform(model["points_xyz"][idxs[in_box]], T)   # Z-up 座標へ
            if ob["dimension_type"] == "height":
                size = float(np.percentile(pts_zup[:, 2], 95) - np.percentile(pts_zup[:, 2], 5))   # 高さ
            else:
                ext = pts_zup[:, :2].max(0) - pts_zup[:, :2].min(0)   # 水平の広がり
                size = float(ext.max())
            if size <= 1e-6:
                continue
            est = ob["typical_dimension_m"] / size   # このオブジェクトからの換算係数
            estimates.append(est)
            details_objs.append({"frame": name, "name": ob["name"], "typ_m": ob["typical_dimension_m"],
                                 "dimension_type": ob["dimension_type"], "colmap_size": size,
                                 "n_points": int(in_box.sum()), "scale": est})
            print(f"  {name}: {ob['name']} 典型 {ob['typical_dimension_m']}m / COLMAP {size:.3f} → 係数 {est:.4f}")

    assert estimates, "GPT 方式で使える既知物体が見つかりませんでした。manual 方式へ切り替えてください。"
    scale = float(np.median(estimates))            # 中央値で頑健化
    print(f"注意: gpt 方式の精度は ±20〜30% です（質量は係数の3乗で効くため誤差が拡大します）")
    return scale, "low", {"estimates": details_objs, "model": llm_model}


# フレームの (幅, 高さ) を返す
def _frame_size(path):
    from PIL import Image                          # 画像サイズ取得用
    with Image.open(path) as img:
        return img.size


def main():
    parser = argparse.ArgumentParser(description="実寸スケール校正（manual/aruco/gpt）")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--mode", choices=["manual", "aruco", "gpt"], required=True, help="校正方式")
    parser.add_argument("--real-dist", type=float, default=None, help="[manual] 実測距離 [m]")
    parser.add_argument("--colmap-dist", type=float, default=None, help="[manual] 3面図から読んだ COLMAP 距離")
    parser.add_argument("--points", default=None, help='[manual] "frame=NAME;a=u,v;b=u,v"')
    parser.add_argument("--marker-size", type=float, default=None, help="[aruco] マーカー一辺の実寸 [m]")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="[aruco] マーカー辞書名")
    args = parser.parse_args()                    # 解析済み引数

    sparse0 = os.path.join(args.workspace, "colmap_ws", "sparse", "0")   # SfM 結果
    frames_dir = os.path.join(args.workspace, "frames")                   # フレーム
    model = load_colmap_model(sparse0)            # COLMAP モデル
    # 3D 点 ID→index 辞書を作って model に添付する（manual/gpt 方式で使用）
    from recon_utils import read_points3d_binary
    _, _, _, id2idx = read_points3d_binary(os.path.join(sparse0, "points3D.bin"))
    model["_id2idx"] = id2idx

    if args.mode == "manual":
        scale, conf, details = mode_manual(args, model)
    elif args.mode == "aruco":
        scale, conf, details = mode_aruco(args, model, frames_dir)
    else:
        scale, conf, details = mode_gpt(args, model, frames_dir, args.workspace)

    assert 1e-4 < scale < 1e4, f"換算係数が異常です: {scale}（入力を確認してください）"
    save_scale(args.workspace, scale, args.mode, conf, details)
    save_stats(args.workspace, "scale", {"scale_factor": scale, "method": args.mode,
                                         "confidence": conf, "details": details})
    print(f"スケール校正完了: 1 COLMAP 単位 = {scale:.4f} m（方式: {args.mode} / 信頼度: {conf}）")
    result_line("SCALE", round(scale, 6))


if __name__ == "__main__":
    main()
