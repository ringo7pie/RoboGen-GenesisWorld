# ============================================================
# build_usd.py
# 役割: 再構成結果（メッシュ / 3DGS / 物性）から Isaac Sim / Omniverse 互換の
#       物理パラメータ付き USD を構築する。
#   - ステージ規約: Z-up / metersPerUnit=1 / kilogramsPerUnit=1 / defaultPrim=/World
#   - 静的環境: 床=UsdGeom.Plane(衝突) + 非床メッシュ=三角形メッシュ衝突（approximation "none"）
#   - 動的剛体 [--phase dynamic]: RigidBodyAPI + MassAPI(質量) + coacd 凸分解 + 物理マテリアル
#   - 3DGS: UsdVol.ParticleField3DGaussianSplat（対応時）/ .ply sidecar + customData（非対応時）
#   - 座標: transform.json（COLMAP→Z-up）と scale.json（実寸換算）をジオメトリへ焼き込む
#   - 付帯出力: Genesis 検証用のベイク済みメッシュ一式（usd/baked/ + manifest.json）
# 使い方: python build_usd.py --workspace <ws> --phase static|dynamic [--gs-mode auto|particlefield|sidecar|none]
#         [--usdz] [--validate] [--usda]
# stdout: RESULT_USD= / RESULT_GS_MODE= / RESULT_VALIDATION=
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # 入力ファイル探索用
import json          # manifest / report の出力用
import os            # パス操作用
import re            # プリム名のサニタイズ用
import shutil        # sidecar コピー用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import (load_gaussian_ply, activate_gaussians, load_transform, load_scale,
                         apply_transform, result_line, save_stats)

# 床の既定物理マテリアル（コンクリート相当。materials_db.yaml の範囲内）
FLOOR_MAT = {"static_friction": 0.6, "dynamic_friction": 0.5, "restitution": 0.2}


# 文字列を USD プリム名として安全な形にする（英数と _ のみ）
def sanitize(name):
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))   # 不正文字を _ に置換
    return ("_" + s) if s and s[0].isdigit() else (s or "prim")


# 4x4 変換 + スケールを点群に適用する（焼き込みの共通処理）
def bake_points(points, T, scale):
    return apply_transform(points, T) * scale


# 回転行列 R を四元数 (w,x,y,z) に変換する
def rotmat_to_quat(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0   # 実部
    if w < 1e-8:                                   # 特異ケースは軸別に計算
        x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
        y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
        z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
        x = np.copysign(x, R[2, 1] - R[1, 2]); y = np.copysign(y, R[0, 2] - R[2, 0])
        z = np.copysign(z, R[1, 0] - R[0, 1])
        return np.array([w, x, y, z])
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


# 四元数の積 q1⊗q2（どちらも wxyz）
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


# trimesh でメッシュを読み、焼き込み済みの (頂点, 面, 頂点色 or None) を返す
def load_baked_mesh(path, T, scale):
    import trimesh                                # メッシュ読込用
    mesh = trimesh.load(path, process=False)      # 属性を保ったまま読む
    verts = bake_points(np.asarray(mesh.vertices), T, scale)   # 焼き込み済み頂点
    faces = np.asarray(mesh.faces)                # 面
    colors = None                                 # 頂点色（あれば 0-1 の float）
    if getattr(mesh.visual, "vertex_colors", None) is not None and len(mesh.visual.vertex_colors):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
    return verts, faces, colors


# ---------------------------------------------------------------
# USD オーサリングの部品
# ---------------------------------------------------------------

# UsdGeom.Mesh を定義して頂点・面・（あれば）頂点色を書き込む
def author_mesh(stage, path, verts, faces, colors=None):
    from pxr import UsdGeom, Vt, Gf, Sdf          # USD ジオメトリ API
    mesh = UsdGeom.Mesh.Define(stage, path)       # メッシュプリム
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)   # ポリゴンのまま（細分割しない）
    ext = np.stack([verts.min(0), verts.max(0)])  # バウンディングボックス
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(ext.astype(np.float32)))
    if colors is not None:
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)   # 頂点色
        pv.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
    return mesh


# 物理マテリアル（摩擦・反発）を UsdShade.Material として定義する
def author_physics_material(stage, path, static_friction, dynamic_friction, restitution):
    from pxr import UsdShade, UsdPhysics          # マテリアル API
    mat = UsdShade.Material.Define(stage, path)   # マテリアルプリム
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())   # 物理属性の付与
    api.CreateStaticFrictionAttr().Set(float(static_friction))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    api.CreateRestitutionAttr().Set(float(restitution))
    return mat


# プリムに物理マテリアルをバインドする（purpose="physics"）
def bind_physics_material(prim, material):
    from pxr import UsdShade                      # バインド API
    binder = UsdShade.MaterialBindingAPI.Apply(prim)
    binder.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


# 3DGS を ParticleField3DGaussianSplat プリムとして書き込む
# 引数: g = activate_gaussians の戻り値、T/scale = 焼き込み変換、max_sh_degree = SH 次数の上限
def author_particlefield(stage, path, g, T, scale, max_sh_degree=3):
    from pxr import UsdVol, Vt, Gf                # ParticleField スキーマ
    pf = UsdVol.ParticleField3DGaussianSplat.Define(stage, path)   # 3DGS プリム
    n = len(g["means"])                           # ガウシアン数

    pos = bake_points(g["means"], T, scale)       # 位置（焼き込み済み）
    R = np.asarray(T)[:3, :3]                     # 変換の回転成分
    qR = rotmat_to_quat(R)                        # 回転四元数（wxyz）
    quats = quat_mul(np.broadcast_to(qR, (n, 4)), g["quats"])   # 各ガウシアンの回転を合成
    scales = g["scales"] * scale                  # スケール（実寸化）

    pf.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    quat_arr = Vt.QuatfArray(n)                   # quatf 配列（real + imaginary）
    for i in range(n):
        quat_arr[i] = Gf.Quatf(float(quats[i, 0]), float(quats[i, 1]), float(quats[i, 2]), float(quats[i, 3]))
    pf.CreateOrientationsAttr(quat_arr)
    pf.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales.astype(np.float32)))
    pf.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(g["opacities"].astype(np.float32)))

    # SH 係数: ガウシアンごとに [dc, rest_1..rest_K] の float3 列を連結（チャネル major の f_rest を並べ替え）
    degree = min(g["sh_degree"], max_sh_degree)   # 実際に書き出す SH 次数
    k_total = (degree + 1) ** 2                   # 1ガウシアンあたりの係数数
    coeffs = np.zeros((n, k_total, 3), dtype=np.float32)   # 係数バッファ
    coeffs[:, 0, :] = g["f_dc"]                   # 0次（DC）
    if degree > 0 and g["f_rest"].shape[1] >= 3 * (k_total - 1):
        n_rest_src = g["f_rest"].shape[1] // 3    # 元データのチャネルあたり高次係数数
        rest = g["f_rest"].reshape(n, 3, n_rest_src)   # [N, ch, K_src]（INRIA はチャネル major）
        coeffs[:, 1:, :] = rest[:, :, :k_total - 1].transpose(0, 2, 1)   # [N, K-1, ch]
    pf.CreateRadianceSphericalHarmonicsCoefficientsAttr(
        Vt.Vec3fArray.FromNumpy(coeffs.reshape(-1, 3)))
    pf.CreateRadianceSphericalHarmonicsDegreeAttr(int(degree))

    ext = np.stack([pos.min(0), pos.max(0)])      # バウンディングボックス
    pf.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(ext.astype(np.float32)))
    return pf, degree


# 3DGS .ply を焼き込み済み sidecar として保存し、参照プリムを書く（ParticleField 非対応時）
def author_sidecar(stage, path, vertex, g, T, scale, out_ply):
    from pxr import UsdGeom                       # Xform プリム用
    from recon_utils import save_gaussian_ply     # ply 書き出し
    baked = vertex.copy()                         # 構造化配列のコピー（全属性維持）
    pos = bake_points(g["means"], T, scale)       # 焼き込み位置
    baked["x"], baked["y"], baked["z"] = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.asarray(T)[:3, :3]                     # 回転成分
    qR = rotmat_to_quat(R)
    quats = quat_mul(np.broadcast_to(qR, (len(g["quats"]), 4)), g["quats"])   # 回転合成
    for i in range(4):
        baked[f"rot_{i}"] = quats[:, i]
    for i in range(3):
        baked[f"scale_{i}"] = np.log(g["scales"][:, i] * scale)   # log スケールへ戻す
    save_gaussian_ply(out_ply, baked)             # sidecar 保存
    xform = UsdGeom.Xform.Define(stage, path)     # 参照用プリム
    xform.GetPrim().SetCustomDataByKey("gaussianSplatPly", os.path.basename(out_ply))
    xform.GetPrim().SetCustomDataByKey("format", "inria-ply")
    xform.GetPrim().SetCustomDataByKey("upAxis", "Z")
    xform.GetPrim().SetCustomDataByKey("metersPerUnit", 1.0)
    return xform


def main():
    parser = argparse.ArgumentParser(description="物理パラメータ付き USD の構築")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--phase", choices=["static", "dynamic"], required=True, help="構築フェーズ")
    parser.add_argument("--gs-mode", choices=["auto", "particlefield", "sidecar", "none"], default="auto",
                        help="3DGS の格納方式（auto=スキーマ有無で自動）")
    parser.add_argument("--gs-sh-degree", type=int, default=3, help="書き出す SH 次数の上限（0で色のみ）")
    parser.add_argument("--usdz", action="store_true", help="usdz パッケージも生成する")
    parser.add_argument("--validate", action="store_true", help="ComplianceChecker で検証する")
    parser.add_argument("--usda", action="store_true", help="デバッグ用にテキスト .usda も出力する")
    args = parser.parse_args()                    # 解析済み引数

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, UsdVol, Gf, Vt   # OpenUSD API

    ws = args.workspace                           # 作業ディレクトリ
    usd_dir = os.path.join(ws, "usd")             # USD 出力先
    baked_dir = os.path.join(usd_dir, "baked")    # Genesis 検証用のベイク済みメッシュ置き場
    os.makedirs(baked_dir, exist_ok=True)

    T = load_transform(ws)                        # COLMAP→Z-up 変換
    scale_info = load_scale(ws)                   # 実寸換算係数
    scale = float(scale_info["scale_factor"])
    if scale_info["method"] == "none":
        print("警告: スケール未校正のため仮係数 1.0 で出力します（セル14 で校正してください）")

    # --- ステージ作成と Isaac 互換メタデータ ---
    usd_path = os.path.join(usd_dir, f"scene_{args.phase}.usdc")   # 出力 USD（バイナリ）
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)   # Z-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)         # 1 unit = 1 m
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)   # 1 unit = 1 kg
    world = UsdGeom.Xform.Define(stage, "/World")     # ルート
    stage.SetDefaultPrim(world.GetPrim())
    Usd.ModelAPI(world.GetPrim()).SetKind("assembly") # kind 設定（Compliance 対策）

    phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")   # 物理シーン
    phys.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))      # 重力方向
    phys.CreateGravityMagnitudeAttr().Set(9.81)                    # 重力加速度 [m/s^2]

    floor_mat = author_physics_material(stage, "/World/PhysicsMaterials/FloorMat", **FLOOR_MAT)

    # --- 静的環境: 床平面 + 非床メッシュ ---
    # 動的フェーズで物体除去済みの残余メッシュがあればそれを使う
    remainder = os.path.join(ws, "objects", "static_remainder.ply")   # 物体除去済みメッシュ
    scene_mesh_src = remainder if (args.phase == "dynamic" and os.path.isfile(remainder)) \
        else os.path.join(ws, "mesh", "scene_mesh.ply")               # 静的メッシュのソース
    verts, faces, colors = load_baked_mesh(scene_mesh_src, T, scale)  # 焼き込み
    ext_min, ext_max = verts.min(0), verts.max(0)                     # シーン範囲（床サイズ決定用）

    env = UsdGeom.Xform.Define(stage, "/World/Environment")           # 静的環境の親
    floor = UsdGeom.Plane.Define(stage, "/World/Environment/Floor")   # 衝突用の床平面（z=0）
    floor.CreateAxisAttr(UsdGeom.Tokens.z)
    floor_size = float(np.linalg.norm(ext_max[:2] - ext_min[:2])) * 2 + 1.0   # 床の一辺
    floor.CreateWidthAttr(floor_size); floor.CreateLengthAttr(floor_size)
    floor.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-floor_size / 2, -floor_size / 2, 0),
                                          Gf.Vec3f(floor_size / 2, floor_size / 2, 0)]))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())                    # 静的コライダ（RigidBody なし）
    bind_physics_material(floor.GetPrim(), floor_mat)

    static_mesh = author_mesh(stage, "/World/Environment/StaticMesh", verts, faces, colors)   # 非床メッシュ
    UsdPhysics.CollisionAPI.Apply(static_mesh.GetPrim())              # 静的コライダ
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(static_mesh.GetPrim())
    mesh_col.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)    # 静的は三角形メッシュのまま
    bind_physics_material(static_mesh.GetPrim(), floor_mat)

    # 床の見た目（再構成された床メッシュ。物理なしの視覚専用）
    floor_visual_src = os.path.join(ws, "mesh", "floor_mesh.ply")     # 床の視覚メッシュ
    if os.path.isfile(floor_visual_src):
        fv, ff, fc = load_baked_mesh(floor_visual_src, T, scale)
        if len(ff) > 0:
            author_mesh(stage, "/World/Environment/FloorVisual", fv, ff, fc)

    # Genesis 検証用に静的メッシュのベイク済み ply を保存
    import trimesh                                 # ベイク保存用
    static_baked_path = os.path.join(baked_dir, "static_scene.ply")   # ベイク済み静的メッシュ
    trimesh.Trimesh(vertices=verts, faces=faces,
                    vertex_colors=None if colors is None else (colors * 255).astype(np.uint8),
                    process=False).export(static_baked_path)
    manifest = {"phase": args.phase, "floor_z": 0.0, "scale_factor": scale,
                "static_mesh": static_baked_path,
                "scene_extent": [ext_min.tolist(), ext_max.tolist()], "objects": []}   # 検証用マニフェスト

    # --- 動的剛体（フェーズ2） ---
    warnings = []                                  # report 用の注意事項
    if args.phase == "dynamic":
        objects_json = os.path.join(ws, "objects", "objects.json")    # 物体一覧
        physics_json = os.path.join(ws, "physics", "physics.json")    # 物性推定
        assert os.path.isfile(objects_json), "objects.json がありません（先に segment_objects.py）"
        assert os.path.isfile(physics_json), "physics.json がありません（先に estimate_physics.py）"
        with open(objects_json, encoding="utf-8") as f:
            objects = json.load(f)["objects"]     # 物体メタ情報
        with open(physics_json, encoding="utf-8") as f:
            physics = json.load(f)["objects"]     # 物性（id キー）

        UsdGeom.Xform.Define(stage, "/World/Objects")   # 動的物体の親
        mats = {}                                  # 材質名→マテリアル（共有）
        for obj in objects:
            oid = str(obj["id"])                   # 物体 ID
            ph = physics.get(oid)                  # 対応する物性
            if ph is None or not ph.get("movable", True):
                warnings.append(f"obj {oid}: 物性なし/固定指定のため静的扱い（StaticMesh に残存）")
                continue
            mesh_path = os.path.join(ws, "objects", f"obj_{oid}", "mesh.ply")   # 視覚メッシュ
            if not os.path.isfile(mesh_path):
                warnings.append(f"obj {oid}: mesh.ply が無いためスキップ")
                continue
            ov, of, oc = load_baked_mesh(mesh_path, T, scale)          # 焼き込み
            centroid = ov.mean(axis=0)             # 剛体原点（Xform の平行移動に使用）
            prim_name = sanitize(f"Obj_{oid}_" + str(obj.get("label", "object")))   # プリム名（Obj_ 前置で数字始まりを回避）
            xpath = f"/World/Objects/{prim_name}"                      # 物体の Xform パス
            xf = UsdGeom.Xform.Define(stage, xpath)
            xf.AddTranslateOp().Set(Gf.Vec3d(*centroid.tolist()))      # 初期位置
            Usd.ModelAPI(xf.GetPrim()).SetKind("component")
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())                # 剛体化
            mass_api = UsdPhysics.MassAPI.Apply(xf.GetPrim())          # 質量（推定値を直接指定）
            mass_api.CreateMassAttr(float(ph["mass"]))

            mat_name = sanitize(ph.get("material", "unknown"))         # 材質名
            if mat_name not in mats:
                mats[mat_name] = author_physics_material(
                    stage, f"/World/PhysicsMaterials/Mat_{mat_name}",
                    ph["static_friction"], ph["dynamic_friction"], ph["restitution"])
            bind_physics_material(xf.GetPrim(), mats[mat_name])

            author_mesh(stage, xpath + "/Render", ov - centroid, of, oc)   # 視覚メッシュ（衝突なし）

            col_scope = UsdGeom.Scope.Define(stage, xpath + "/Collision")  # 衝突形状の親
            UsdGeom.Imageable(col_scope.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)   # 非表示
            col_files = sorted(glob.glob(os.path.join(ws, "objects", f"obj_{oid}", "collision", "*.obj")))
            if not col_files:
                col_files = [mesh_path]            # 凸分解が無ければ視覚メッシュの凸包を使う
            for ci, cf in enumerate(col_files):
                cv, cfaces, _ = load_baked_mesh(cf, T, scale)          # 凸パーツの焼き込み
                cmesh = author_mesh(stage, xpath + f"/Collision/part_{ci:02d}", cv - centroid, cfaces)
                UsdPhysics.CollisionAPI.Apply(cmesh.GetPrim())
                capprox = UsdPhysics.MeshCollisionAPI.Apply(cmesh.GetPrim())
                capprox.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)   # 動的は凸近似必須
                bind_physics_material(cmesh.GetPrim(), mats[mat_name])

            # Genesis 検証用ベイク + マニフェスト登録
            obj_baked = os.path.join(baked_dir, f"obj_{oid}.ply")      # ベイク済み視覚メッシュ
            trimesh.Trimesh(vertices=ov, faces=of,
                            vertex_colors=None if oc is None else (oc * 255).astype(np.uint8),
                            process=False).export(obj_baked)
            manifest["objects"].append({
                "id": oid, "label": obj.get("label", "object"), "visual": obj_baked,
                "mass": float(ph["mass"]), "material": ph.get("material", "unknown"),
                "static_friction": float(ph["static_friction"]),
                "restitution": float(ph["restitution"]), "centroid": centroid.tolist()})

    # --- 3DGS の格納 ---
    gs_mode = args.gs_mode                        # 実際に使う格納方式
    if gs_mode == "auto":
        gs_mode = "particlefield" if hasattr(UsdVol, "ParticleField3DGaussianSplat") else "sidecar"
    sh_written = None                             # 書き出した SH 次数（report 用）
    if gs_mode != "none":
        plys = sorted(glob.glob(os.path.join(ws, "gsplat", "ply", "*.ply")))   # 学習済み 3DGS
        assert plys, "3DGS の .ply がありません（先に train_gsplat.py）"
        vertex, _ = load_gaussian_ply(plys[-1])   # 構造化配列
        g = activate_gaussians(vertex)            # 活性化済み
        UsdGeom.Xform.Define(stage, "/World/GaussianSplats")           # 3DGS の親
        if gs_mode == "particlefield":
            _, sh_written = author_particlefield(stage, "/World/GaussianSplats/SceneSplats",
                                                 g, T, scale, args.gs_sh_degree)
            warnings.append("SH 係数は座標回転に対して未回転のまま格納（視点依存色は近似）")
        else:
            author_sidecar(stage, "/World/GaussianSplats/SceneSplats",
                           vertex, g, T, scale, os.path.join(usd_dir, "gaussians.ply"))

    stage.GetRootLayer().Save()                   # .usdc 保存
    print(f"USD 保存: {usd_path}")
    if args.usda:
        usda_path = usd_path.replace(".usdc", ".usda")   # デバッグ用テキスト出力
        stage.GetRootLayer().Export(usda_path)
        print(f"usda 保存: {usda_path}")

    # --- usdz パッケージ化 ---
    usdz_path = None                              # 生成した usdz のパス
    if args.usdz:
        usdz_path = os.path.join(usd_dir, "scene.usdz")
        ok = UsdUtils.CreateNewUsdzPackage(usd_path, usdz_path)
        print(f"usdz 生成: {usdz_path}（{'成功' if ok else '失敗'}）")
        if gs_mode == "sidecar":
            warnings.append("sidecar の gaussians.ply は usdz に同梱されません（usd と同じフォルダで配布してください）")

    # --- ComplianceChecker 検証 ---
    validation = "SKIPPED"                        # 検証結果の要約
    val_details = {}                              # 詳細
    if args.validate:
        checker = UsdUtils.ComplianceChecker()    # 標準の適合性チェッカ
        checker.CheckCompliance(usd_path)
        errors = list(checker.GetErrors()) + list(checker.GetFailedChecks())   # エラーと失敗項目
        val_details = {"errors": errors, "warnings": [str(w) for w in checker.GetWarnings()]}
        validation = "PASS" if not errors else f"FAIL({len(errors)})"
        print(f"Compliance: {validation}")
        for e in errors:
            print("  -", e)

    # --- マニフェストと report の保存 ---
    with open(os.path.join(baked_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    report = {"usd": usd_path, "usdz": usdz_path, "phase": args.phase, "gs_mode": gs_mode,
              "sh_degree_written": sh_written, "scale": scale_info, "validation": validation,
              "validation_details": val_details, "warnings": warnings,
              "num_objects": len(manifest["objects"])}
    with open(os.path.join(usd_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    save_stats(ws, f"usd_{args.phase}", report)

    result_line("USD", usd_path)
    result_line("GS_MODE", gs_mode)
    result_line("VALIDATION", validation)


if __name__ == "__main__":
    main()
