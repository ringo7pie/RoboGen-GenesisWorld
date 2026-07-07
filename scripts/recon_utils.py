# ============================================================
# recon_utils.py
# 役割: ノートブック③（Real2Sim パイプライン）の共有ユーティリティ。
#   - 設定読み込み（recon_defaults.yaml + プリセットマージ）
#   - COLMAP バイナリモデル (cameras/images/points3D.bin) の純 Python パーサ
#     （公式 pycolmap / rmbrualla 版のどちらにも依存しない = 名前衝突の影響を受けない）
#   - 3DGS .ply（INRIA 形式）の読み書きと活性化（exp/sigmoid/正規化）
#   - gsplat による深度+RGB レンダリングと Open3D TSDF 融合
#   - transform.json / scale.json の入出力、RESULT_ 行・stats JSON の出力
# 依存: numpy, plyfile, pyyaml（レンダ系関数のみ torch/gsplat、TSDF のみ open3d）
# ============================================================
import json          # stats / transform / scale の入出力用
import os            # パス操作用
import struct        # COLMAP バイナリの読み取り用
import numpy as np   # 数値計算用


# ---------------------------------------------------------------
# 設定・出力ヘルパ
# ---------------------------------------------------------------

# recon_defaults.yaml を読み、プリセット（indoor/outdoor）を上書きマージして返す
# 引数: preset = プリセット名（None なら既定値のまま）
# 戻り値: 設定辞書
def load_recon_config(preset=None):
    import yaml                                   # 設定ファイルの解析用
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "configs", "recon_defaults.yaml")   # 既定設定のパス
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)                   # 既定設定
    if preset and preset in cfg.get("presets", {}):
        for section, overrides in cfg["presets"][preset].items():
            cfg.setdefault(section, {}).update(overrides)   # プリセット値で上書き
    return cfg


# ノートブックがパースする機械可読行（RESULT_KEY=value）を出力する
def result_line(key, value):
    print(f"RESULT_{key}={value}")


# ステージの統計情報を <workspace>/<stage>_stats.json に保存する（エラー報告の1次情報）
def save_stats(workspace, stage, stats):
    os.makedirs(workspace, exist_ok=True)
    path = os.path.join(workspace, f"{stage}_stats.json")   # 保存先
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"stats 保存: {path}")
    return path


# ---------------------------------------------------------------
# COLMAP バイナリモデルの純 Python パーサ
# フォーマット仕様: https://colmap.github.io/format.html
# ---------------------------------------------------------------

# COLMAP カメラモデル ID → (名前, パラメータ数) の対応表
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


# cameras.bin を読み、{camera_id: {model, width, height, params}} を返す
def read_cameras_binary(path):
    cameras = {}                                  # 読み取り結果
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]   # カメラ数
        for _ in range(num):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            name, n_params = _CAMERA_MODELS[model_id]                     # モデル名とパラメータ数
            params = np.array(struct.unpack("<" + "d" * n_params, f.read(8 * n_params)))
            cameras[cam_id] = {"model": name, "width": int(width), "height": int(height), "params": params}
    return cameras


# images.bin を読み、{image_id: {qvec, tvec, camera_id, name, xys, point3D_ids}} を返す
# qvec/tvec は world→camera（COLMAP 規約）
def read_images_binary(path):
    images = {}                                   # 読み取り結果
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]   # 画像数
        for _ in range(num):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))   # 回転（w,x,y,z）
            tvec = np.array(struct.unpack("<ddd", f.read(24)))    # 並進
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""                            # 画像ファイル名（NUL 終端）
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            n_pts = struct.unpack("<Q", f.read(8))[0]             # 2D 特徴点数
            data = f.read(24 * n_pts)                              # (x, y, point3D_id) の列
            arr = np.frombuffer(data, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id,
                                "name": name.decode("utf-8"),
                                "xys": np.stack([arr["x"], arr["y"]], axis=1),
                                "point3D_ids": arr["id"].copy()}
    return images


# points3D.bin を読み、(xyz[N,3], rgb[N,3], error[N], id→index 辞書) を返す
def read_points3d_binary(path):
    xyzs, rgbs, errors, id2idx = [], [], [], {}   # 読み取りバッファ
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]   # 3D 点数
        for i in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]          # トラック長
            f.read(8 * track_len)                                   # トラック本体は読み飛ばす
            xyzs.append(xyz); rgbs.append(rgb); errors.append(err); id2idx[pid] = i
    return np.array(xyzs), np.array(rgbs, dtype=np.uint8), np.array(errors), id2idx


# COLMAP の qvec（w,x,y,z）を 3x3 回転行列に変換する
def qvec_to_rotmat(qvec):
    w, x, y, z = qvec                             # 四元数の成分
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


# sparse モデル一式を読み、レンダリングに使いやすい形へ整理して返す
# 戻り値: {"cameras", "images", "points_xyz", "points_rgb",
#          "views": [ {name, K(3x3), w2c(4x4), width, height, image_id} ... ]}
def load_colmap_model(sparse_dir):
    cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    pts_xyz, pts_rgb, pts_err, _ = read_points3d_binary(os.path.join(sparse_dir, "points3D.bin"))

    views = []                                    # ビュー（画像）ごとのカメラ情報
    for image_id, im in sorted(images.items(), key=lambda kv: kv[1]["name"]):
        cam = cameras[im["camera_id"]]            # 対応するカメラ
        fx, fy, cx, cy = _pinhole_params(cam)     # ピンホール近似の内部パラメータ
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])   # 内部行列
        w2c = np.eye(4)                           # world→camera 変換
        w2c[:3, :3] = qvec_to_rotmat(im["qvec"])
        w2c[:3, 3] = im["tvec"]
        views.append({"name": im["name"], "K": K, "w2c": w2c,
                      "width": cam["width"], "height": cam["height"], "image_id": image_id})
    return {"cameras": cameras, "images": images,
            "points_xyz": pts_xyz, "points_rgb": pts_rgb, "points_err": pts_err, "views": views}


# カメラモデルからピンホール近似の (fx, fy, cx, cy) を取り出す（歪み係数は無視 = 近似）
def _pinhole_params(cam):
    p = cam["params"]                             # モデル依存のパラメータ列
    if cam["model"] == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    if cam["model"] == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if cam["model"] in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        return p[0], p[0], p[1], p[2]
    # OPENCV 系: fx, fy, cx, cy が先頭4つ
    return p[0], p[1], p[2], p[3]


# ---------------------------------------------------------------
# 3DGS .ply（INRIA 形式）の読み書き
# 頂点属性: x,y,z / f_dc_0..2 / f_rest_* / opacity(logit) / scale_0..2(log) / rot_0..3(wxyz)
# ---------------------------------------------------------------

# .ply を構造化配列のまま読み込む（属性を欠落させずサブセット保存できる形）
# 戻り値: (numpy 構造化配列, 属性名リスト)
def load_gaussian_ply(path):
    from plyfile import PlyData                   # PLY パーサ
    ply = PlyData.read(path)                      # 読み込み
    v = ply["vertex"].data                        # 頂点（=ガウシアン）配列
    return np.asarray(v), [p.name for p in ply["vertex"].properties]


# 構造化配列（load_gaussian_ply の戻り値と同形式）を .ply として保存する
def save_gaussian_ply(path, vertex_array):
    from plyfile import PlyData, PlyElement       # PLY 書き出し
    el = PlyElement.describe(vertex_array, "vertex")   # 頂点要素
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    PlyData([el], text=False).write(path)         # バイナリ PLY で保存


# 構造化配列から活性化済みパラメータを取り出す
# 戻り値: dict(means[N,3], quats[N,4](正規化, wxyz), scales[N,3](exp 済),
#              opacities[N](sigmoid 済), colors[N,3](SH0→RGB), sh_degree, f_dc[N,3], f_rest[N,R])
def activate_gaussians(v):
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)          # 中心座標
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64) # 回転（wxyz）
    quats /= (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)                # 正規化
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))  # スケール（log→実寸）
    opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))                  # 不透明度（logit→[0,1]）
    f_dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float64) # SH 0次係数
    rest_names = sorted([n for n in v.dtype.names if n.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))                       # SH 高次係数の属性名
    f_rest = (np.stack([v[n] for n in rest_names], axis=1).astype(np.float64)
              if rest_names else np.zeros((len(v), 0)))
    n_rest_per_ch = len(rest_names) // 3 if rest_names else 0                      # チャネルあたりの高次係数数
    degree = {0: 0, 3: 1, 8: 2, 15: 3}.get(n_rest_per_ch, 0)                       # SH 次数
    colors = np.clip(0.28209479177387814 * f_dc + 0.5, 0.0, 1.0)                   # SH0 → RGB
    return {"means": means, "quats": quats, "scales": scales, "opacities": opac,
            "colors": colors, "f_dc": f_dc, "f_rest": f_rest, "sh_degree": degree}


# ---------------------------------------------------------------
# gsplat 深度レンダリング + Open3D TSDF 融合（extract_mesh / segment_objects で使用）
# ---------------------------------------------------------------

# 活性化済みガウシアンを torch テンソル化して GPU に載せる
# 戻り値: gsplat.rasterization にそのまま渡せる辞書
def gaussians_to_torch(g, device="cuda"):
    import torch                                  # GPU テンソル化用
    return {
        "means": torch.tensor(g["means"], dtype=torch.float32, device=device),
        "quats": torch.tensor(g["quats"], dtype=torch.float32, device=device),
        "scales": torch.tensor(g["scales"], dtype=torch.float32, device=device),
        "opacities": torch.tensor(g["opacities"], dtype=torch.float32, device=device),
        "colors": torch.tensor(g["colors"], dtype=torch.float32, device=device),
    }


# 1視点分の RGB+深度をレンダリングする（render_mode="RGB+ED" の期待深度を使用）
# 引数: gt = gaussians_to_torch の戻り値、w2c = 4x4 world→camera、K = 3x3、width/height = 出力解像度
# 戻り値: (rgb[H,W,3] float32 0-1, depth[H,W] float32)
def render_rgb_depth(gt, w2c, K, width, height):
    import torch                                  # テンソル演算用
    from gsplat import rasterization              # 3DGS ラスタライザ
    device = gt["means"].device                   # 実行デバイス
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)[None]   # [1,4,4]
    Kt = torch.tensor(K, dtype=torch.float32, device=device)[None]          # [1,3,3]
    with torch.no_grad():
        out, alpha, _ = rasterization(
            gt["means"], gt["quats"], gt["scales"], gt["opacities"], gt["colors"],
            viewmat, Kt, int(width), int(height),
            render_mode="RGB+ED",                 # RGB + 期待深度（最終チャネル）
        )
    img = out[0]                                  # [H,W,4]
    rgb = img[..., :3].clamp(0, 1).cpu().numpy()  # RGB 部分
    depth = img[..., 3].cpu().numpy()             # 期待深度
    alpha_np = alpha[0, ..., 0].cpu().numpy()     # 累積不透明度（未カバー領域の除外用）
    depth[alpha_np < 0.5] = 0.0                   # ガウシアンが無い画素は深度 0（TSDF が無視する）
    return rgb.astype(np.float32), depth.astype(np.float32)


# 複数視点の RGB-D を Open3D の TSDF ボリュームへ融合し、三角形メッシュを返す
# 引数: gt = GPU 上のガウシアン、views = load_colmap_model の views、
#       voxel_size = ボクセル寸法（COLMAP 単位）、depth_trunc = 深度打ち切り、
#       render_scale = レンダ解像度の縮小率（速度対策）
def tsdf_fuse(gt, views, voxel_size, depth_trunc, render_scale=0.5, log_every=20):
    import open3d as o3d                          # TSDF 融合用
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_size),           # ボクセル寸法
        sdf_trunc=float(voxel_size) * 4.0,        # SDF 打ち切り幅（慣例: ボクセルの数倍）
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i, view in enumerate(views):
        w = max(64, int(view["width"] * render_scale))    # レンダ幅
        h = max(64, int(view["height"] * render_scale))   # レンダ高
        K = view["K"].copy()                       # 内部行列（解像度に合わせてスケール）
        K[0] *= w / view["width"]
        K[1] *= h / view["height"]
        rgb, depth = render_rgb_depth(gt, view["w2c"], K, w, h)   # RGB-D レンダ
        color_img = o3d.geometry.Image((rgb * 255).astype(np.uint8))
        depth_img = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0,        # depth は実距離そのまま
            depth_trunc=float(depth_trunc), convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        volume.integrate(rgbd, intrinsic, view["w2c"])    # extrinsic は world→camera
        if (i + 1) % log_every == 0:
            print(f"TSDF 統合: {i + 1}/{len(views)} 視点")
    return volume.extract_triangle_mesh()          # 融合結果のメッシュ


# ---------------------------------------------------------------
# 座標変換（COLMAP → 床基準 Z-up）と実寸スケール
# ---------------------------------------------------------------

# transform.json（COLMAP→Z-up の 4x4 行列）を保存する
def save_transform(workspace, matrix, meta=None):
    path = os.path.join(workspace, "mesh", "transform.json")   # 保存先
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"colmap_to_zup": np.asarray(matrix).tolist(), "meta": meta or {}}, f, indent=2)
    return path


# transform.json を読み込む（無ければ単位行列）
def load_transform(workspace):
    path = os.path.join(workspace, "mesh", "transform.json")
    if not os.path.isfile(path):
        return np.eye(4)
    with open(path, encoding="utf-8") as f:
        return np.array(json.load(f)["colmap_to_zup"])


# scale.json（実寸換算係数）を保存する
def save_scale(workspace, scale_factor, method, confidence, details=None):
    path = os.path.join(workspace, "scale", "scale.json")      # 保存先
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scale_factor": float(scale_factor), "method": method,
                   "confidence": confidence, "details": details or {}}, f, indent=2, ensure_ascii=False)
    return path


# scale.json を読み込む（無ければ係数 1.0 の仮スケール）
def load_scale(workspace):
    path = os.path.join(workspace, "scale", "scale.json")
    if not os.path.isfile(path):
        return {"scale_factor": 1.0, "method": "none", "confidence": "low", "details": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 4x4 同次変換を点群 [N,3] に適用する
def apply_transform(points, matrix):
    pts = np.asarray(points)                      # 入力点群
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)   # 同次座標化
    return (homo @ np.asarray(matrix).T)[:, :3]
