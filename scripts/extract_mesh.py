# ============================================================
# extract_mesh.py
# 役割: 学習済み 3DGS から衝突用の三角形メッシュを抽出し、床平面を分離する
#   ① .ply ロード + 不透明度フィルタ（フロータ除去）
#   ② 全学習視点で RGB+深度をレンダ → Open3D TSDF 融合 → marching cubes
#   ③ 小クラスタ除去 + 簡略化（decimation）
#   ④ RANSAC で床平面を検出し、COLMAP→「床=z0・+Z上」の変換行列を算出
#   ⑤ 床/非床メッシュの分割保存 + 目盛り付き3面図 PNG（スケール校正 manual の資料）
# 使い方: python extract_mesh.py --workspace <RUN_DIR> [--preset indoor|outdoor]
# 出力: <ws>/mesh/{fused_mesh.ply, scene_mesh.ply, floor_mesh.ply,
#                  floor_plane.json, transform.json, preview_{top,front,side}.png}
# stdout: RESULT_MESH= / RESULT_TRIANGLES= / RESULT_FLOOR_INLIER_RATIO=
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # 入力 ply の探索用
import json          # floor_plane.json の出力用
import os            # パス操作用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import (load_recon_config, load_colmap_model, load_gaussian_ply,
                         activate_gaussians, gaussians_to_torch, tsdf_fuse,
                         result_line, save_stats, save_transform, apply_transform)


# 平面法線 n を +Z へ回す回転行列を返す（Rodrigues の回転公式）
def rotation_to_z(n):
    n = np.asarray(n, dtype=np.float64)
    n /= np.linalg.norm(n)                        # 正規化した法線
    z = np.array([0.0, 0.0, 1.0])                 # 目標方向
    v = np.cross(n, z)                            # 回転軸（未正規化）
    c = float(np.dot(n, z))                       # cos(回転角)
    if np.linalg.norm(v) < 1e-8:                  # 既に ±Z を向いている場合
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])   # 外積行列
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def main():
    parser = argparse.ArgumentParser(description="3DGS からの衝突メッシュ抽出 + 床分離")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--preset", choices=["indoor", "outdoor"], default="indoor", help="シーンプリセット")
    parser.add_argument("--max-views", type=int, default=120, help="TSDF に使う最大視点数（間引き）")
    parser.add_argument("--render-scale", type=float, default=0.5, help="深度レンダの解像度縮小率")
    parser.add_argument("--force", action="store_true", help="既存出力を破棄して再実行")
    args = parser.parse_args()                    # 解析済み引数

    cfg = load_recon_config(args.preset)["mesh"]  # プリセット適用済みの既定値
    mesh_dir = os.path.join(args.workspace, "mesh")               # 出力先
    scene_mesh_path = os.path.join(mesh_dir, "scene_mesh.ply")    # 非床メッシュ（頂点色付き）
    if os.path.isfile(scene_mesh_path) and not args.force:
        print(f"既存のメッシュを再利用します: {scene_mesh_path}（やり直す場合は --force）")
        result_line("MESH", scene_mesh_path)
        return
    os.makedirs(mesh_dir, exist_ok=True)

    import open3d as o3d                          # メッシュ処理（重いので main 内 import）

    # --- ① 3DGS ロード + フロータ除去 ---
    plys = sorted(glob.glob(os.path.join(args.workspace, "gsplat", "ply", "*.ply")))   # 学習済み ply
    assert plys, "学習済み .ply がありません（先に train_gsplat.py を実行）"
    vertex, _ = load_gaussian_ply(plys[-1])       # 構造化配列
    g = activate_gaussians(vertex)                # 活性化済みパラメータ
    keep = g["opacities"] >= cfg["opacity_min"]   # 不透明度フィルタ
    for k in ("means", "quats", "scales", "opacities", "colors"):
        g[k] = g[k][keep]
    print(f"ガウシアン: {len(keep)} → {keep.sum()}（opacity>={cfg['opacity_min']} で除去）")

    # --- シーン規模の自動決定（ガウシアン中心の 5-95% パーセンタイル bbox） ---
    lo = np.percentile(g["means"], 5, axis=0)     # bbox 下限
    hi = np.percentile(g["means"], 95, axis=0)    # bbox 上限
    scene_diag = float(np.linalg.norm(hi - lo))   # シーン対角（COLMAP 単位）
    voxel = scene_diag / cfg["voxel_ratio"]       # TSDF ボクセル寸法

    # --- ② TSDF 融合 ---
    model = load_colmap_model(os.path.join(args.workspace, "colmap_ws", "sparse", "0"))   # カメラ情報
    views = model["views"]                        # 全視点
    if len(views) > args.max_views:
        step = len(views) / args.max_views        # 間引き間隔
        views = [views[int(i * step)] for i in range(args.max_views)]
    cam_pos = np.stack([np.linalg.inv(v["w2c"])[:3, 3] for v in views])   # カメラ位置（world）
    cam_diag = float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0)) + 1e-6)   # カメラ軌跡の広がり
    depth_trunc = max(cam_diag, scene_diag * 0.5) * cfg["depth_trunc_factor"]  # 深度打ち切り（遠景対策）
    print(f"シーン対角={scene_diag:.2f} voxel={voxel:.4f} depth_trunc={depth_trunc:.2f}（COLMAP 単位）")

    gt = gaussians_to_torch(g)                    # GPU テンソル化
    mesh = tsdf_fuse(gt, views, voxel, depth_trunc, render_scale=args.render_scale)   # TSDF → メッシュ
    print(f"TSDF メッシュ: 頂点 {len(mesh.vertices)} / 三角形 {len(mesh.triangles)}")
    assert len(mesh.triangles) > 1000, "メッシュがほぼ空です。学習品質か depth_trunc を確認してください。"
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, "fused_mesh.ply"), mesh)   # 融合直後を保存

    # --- ③ 小クラスタ除去 + 簡略化 ---
    tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()   # 連結成分解析
    tri_clusters = np.asarray(tri_clusters); cluster_n_tri = np.asarray(cluster_n_tri)
    largest = cluster_n_tri.max()                 # 最大クラスタの三角形数
    remove_mask = cluster_n_tri[tri_clusters] < largest * cfg["min_cluster_ratio"]   # 小クラスタ判定
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > cfg["decimate_target"]:
        mesh = mesh.simplify_quadric_decimation(cfg["decimate_target"])   # 簡略化
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    print(f"クリーニング後: 頂点 {len(mesh.vertices)} / 三角形 {len(mesh.triangles)}")

    # --- ④ 床平面の検出（RANSAC）と Z-up 変換の算出 ---
    pcd = o3d.geometry.PointCloud(mesh.vertices)  # 頂点を点群として平面検出
    dist_th = scene_diag * cfg["floor_dist_ratio"]                 # 平面距離閾値
    plane, inliers = pcd.segment_plane(distance_threshold=dist_th, ransac_n=3, num_iterations=2000)
    a, b, c_, d = plane                           # 平面 ax+by+cz+d=0
    n = np.array([a, b, c_])                      # 平面法線
    inlier_ratio = len(inliers) / len(mesh.vertices)               # 床 inlier 率

    # カメラの「上」方向（COLMAP はカメラ y 軸が画像下向き → world 上 ≈ -R^T e_y の平均）と符号合わせ
    ups = np.stack([-v["w2c"][:3, :3].T @ np.array([0, 1, 0]) for v in views])   # 各視点の上方向
    avg_up = ups.mean(axis=0)                     # 平均上方向
    if float(np.dot(n, avg_up)) < 0:              # 法線が下向きなら反転
        n, d = -n, -d
    R = rotation_to_z(n)                          # 法線→+Z の回転
    verts = np.asarray(mesh.vertices)             # メッシュ頂点
    p0 = verts[inliers].mean(axis=0)              # 床上の代表点
    T = np.eye(4)                                 # COLMAP→Z-up 変換
    T[:3, :3] = R
    T[:3, 3] = -R @ p0                            # 代表点が原点（z=0 床）に来る平行移動
    save_transform(args.workspace, T, meta={"plane": [float(a), float(b), float(c_), float(d)],
                                            "inlier_ratio": float(inlier_ratio),
                                            "scene_diag_colmap": scene_diag})
    print(f"床平面: inlier率 {inlier_ratio:.0%} / 距離閾値 {dist_th:.4f}")

    # --- ⑤ 床/非床の分割保存 ---
    vert_dist = np.abs(verts @ n / np.linalg.norm(n) + d / np.linalg.norm(n))   # 各頂点の平面距離
    floor_vert = vert_dist < dist_th * 1.5        # 床とみなす頂点
    tris = np.asarray(mesh.triangles)             # 三角形インデックス
    tri_floor = floor_vert[tris].all(axis=1)      # 3頂点すべて床なら床三角形
    floor_mesh = o3d.geometry.TriangleMesh(mesh)  # 床メッシュ（コピーから抽出）
    floor_mesh.remove_triangles_by_mask(~tri_floor)
    floor_mesh.remove_unreferenced_vertices()
    scene_mesh = o3d.geometry.TriangleMesh(mesh)  # 非床メッシュ
    scene_mesh.remove_triangles_by_mask(tri_floor)
    scene_mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, "floor_mesh.ply"), floor_mesh)
    o3d.io.write_triangle_mesh(scene_mesh_path, scene_mesh)
    with open(os.path.join(mesh_dir, "floor_plane.json"), "w", encoding="utf-8") as f:
        json.dump({"normal": n.tolist(), "d": float(d), "inlier_ratio": float(inlier_ratio)}, f, indent=2)

    # --- 目盛り付き3面図（スケール校正 manual 方式の読み取り資料） ---
    _save_previews(mesh_dir, apply_transform(verts, T))

    save_stats(args.workspace, "mesh", {
        "ply_source": plys[-1], "triangles": len(scene_mesh.triangles),
        "floor_triangles": len(floor_mesh.triangles), "floor_inlier_ratio": float(inlier_ratio),
        "voxel": voxel, "depth_trunc": depth_trunc, "scene_diag_colmap": scene_diag,
        "views_used": len(views), "preset": args.preset})
    result_line("MESH", scene_mesh_path)
    result_line("TRIANGLES", len(scene_mesh.triangles))
    result_line("FLOOR_INLIER_RATIO", round(inlier_ratio, 3))


# Z-up 変換済み頂点の3面図（top/front/side）を COLMAP 単位の目盛り付きで保存する
def _save_previews(mesh_dir, verts_zup):
    import matplotlib                              # ヘッドレス描画用
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = verts_zup[np.random.choice(len(verts_zup), min(60000, len(verts_zup)), replace=False)]   # 描画用サンプル
    axes_def = [("top", 0, 1, "X", "Y"), ("front", 0, 2, "X", "Z"), ("side", 1, 2, "Y", "Z")]   # 3面図の定義
    for name, i, j, xl, yl in axes_def:
        fig, ax = plt.subplots(figsize=(7, 7))    # 1面図
        ax.scatter(sample[:, i], sample[:, j], s=0.3, c="#3565c0", alpha=0.5, linewidths=0)
        ax.set_aspect("equal")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.6)
        ax.minorticks_on()
        ax.set_xlabel(f"{xl} [COLMAP 単位]"); ax.set_ylabel(f"{yl} [COLMAP 単位]")
        ax.set_title(f"{name} view（目盛りから既知物の寸法を読み取り、スケール校正 manual に使用）")
        fig.tight_layout()
        fig.savefig(os.path.join(mesh_dir, f"preview_{name}.png"), dpi=110)
        plt.close(fig)
    print(f"3面図を保存: {mesh_dir}/preview_{{top,front,side}}.png")


if __name__ == "__main__":
    main()
