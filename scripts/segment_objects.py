# ============================================================
# segment_objects.py
# 役割: テキストプロンプトで指定した物体を 3DGS シーンから分離する（フェーズ2）
#   ① 代表視点に GroundingDINO（テキスト→検出枠）+ SAM2（枠→マスク）を適用
#   ② ガウシアン中心を各視点へ投影し「マスク内に入った率」で投票 → 物体別ガウシアン集合
#   ③ 物体ガウシアンのみで TSDF → 物体メッシュ、coacd で凸分解（衝突形状）
#   ④ シーンメッシュから物体近傍の三角形を除いた static_remainder.ply を生成
# 使い方: python segment_objects.py --workspace <ws> --prompts "chair. table. box."
# 出力: <ws>/objects/{objects.json, static_remainder.ply, obj_<id>/{gaussians.ply, mesh.ply,
#        collision/part_*.obj, crop.jpg}}
# stdout: RESULT_OBJECTS= / RESULT_NUM_OBJECTS=
# ============================================================
import argparse      # コマンドライン引数の解析用
import glob          # フレーム探索用
import json          # objects.json の出力用
import os            # パス操作用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import (load_recon_config, load_colmap_model, load_gaussian_ply,
                         activate_gaussians, save_gaussian_ply, gaussians_to_torch,
                         tsdf_fuse, result_line, save_stats)


# GroundingDINO + SAM2 を1画像に適用し、[(label, score, box_xyxy, mask[H,W]bool)] を返す
def detect_and_segment(image, prompts, dino, dino_proc, sam2, sam2_proc, device, box_th, text_th):
    import torch                                  # 推論用
    inputs = dino_proc(images=image, text=prompts, return_tensors="pt").to(device)   # DINO 入力
    with torch.no_grad():
        outputs = dino(**inputs)
    H, W = image.height, image.width              # 画像サイズ
    res = dino_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_th, text_threshold=text_th,
        target_sizes=[(H, W)])[0]                 # 検出結果（boxes/scores/labels）
    dets = []                                     # 検出リスト
    for box, score, label in zip(res["boxes"], res["scores"], res.get("text_labels", res["labels"])):
        box_list = [[[float(v) for v in box.tolist()]]]   # SAM2 の input_boxes 形式 [batch][obj][4]
        sinp = sam2_proc(images=image, input_boxes=box_list, return_tensors="pt").to(device)   # SAM2 入力
        with torch.no_grad():
            sout = sam2(**sinp, multimask_output=False)
        masks = sam2_proc.post_process_masks(sout.pred_masks.cpu(),
                                             sinp["original_sizes"].cpu())[0]   # [n,1,H,W]
        mask = masks[0, 0].numpy() > 0.5          # bool マスク
        dets.append((str(label), float(score), [float(v) for v in box.tolist()], mask))
    return dets


# ガウシアン中心を1視点へ投影し、(uv[N,2], 前方フラグ[N]) を返す
def project_points(means, view):
    w2c = view["w2c"]                             # world→camera
    cam = means @ w2c[:3, :3].T + w2c[:3, 3]      # カメラ座標
    in_front = cam[:, 2] > 1e-6                   # カメラ前方の点のみ有効
    uv = cam @ view["K"].T                        # 射影（同次）
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)   # 画素座標
    return uv, in_front


def main():
    parser = argparse.ArgumentParser(description="テキスト指定の物体分離（GroundingDINO+SAM2+投票）")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--prompts", required=True, help='検出したい物体（例 "chair. table. box."）')
    parser.add_argument("--views", type=int, default=12, help="使用する視点数")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="DINO の検出閾値")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="DINO のテキスト閾値")
    parser.add_argument("--min-votes", type=float, default=0.6, help="採用に必要な投票率（観測視点に対する）")
    parser.add_argument("--min-gaussians", type=int, default=200, help="物体として扱う最小ガウシアン数")
    parser.add_argument("--max-objects", type=int, default=10, help="最大物体数")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base", help="DINO のモデル ID")
    parser.add_argument("--sam2-model", default="facebook/sam2-hiera-small", help="SAM2 のモデル ID")
    args = parser.parse_args()                    # 解析済み引数

    import torch                                  # 推論デバイス判定用
    from PIL import Image                         # フレーム読み込み用
    from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                              Sam2Model, Sam2Processor)   # HF 実装（公式リポジトリ clone 不要）

    device = "cuda" if torch.cuda.is_available() else "cpu"   # 推論デバイス
    print(f"モデルをロード中（{device}）: {args.dino_model} / {args.sam2_model}")
    dino_proc = AutoProcessor.from_pretrained(args.dino_model)          # DINO 前処理
    dino = GroundingDinoForObjectDetection.from_pretrained(args.dino_model).to(device).eval()
    sam2_proc = Sam2Processor.from_pretrained(args.sam2_model)          # SAM2 前処理
    sam2 = Sam2Model.from_pretrained(args.sam2_model).to(device).eval()

    ws = args.workspace                           # 作業ディレクトリ
    obj_root = os.path.join(ws, "objects")        # 出力先
    os.makedirs(obj_root, exist_ok=True)

    plys = sorted(glob.glob(os.path.join(ws, "gsplat", "ply", "*.ply")))   # 学習済み 3DGS
    vertex, _ = load_gaussian_ply(plys[-1])       # 構造化配列（全属性保持）
    g = activate_gaussians(vertex)                # 活性化済み
    means = g["means"]                            # ガウシアン中心

    model = load_colmap_model(os.path.join(ws, "colmap_ws", "sparse", "0"))   # カメラ情報
    views_all = model["views"]                    # 全視点
    step = max(1, len(views_all) // args.views)   # 視点間引きの間隔
    views = views_all[::step][:args.views]        # 使用する視点
    frames_dir = os.path.join(ws, "frames")       # フレーム画像の場所

    # --- 検出→投票（視点をまたいだ物体対応付けはガウシアン集合の IoU で貪欲マージ） ---
    objects = []                                  # [{label_counts, votes(np.int32[N]), n_obs, best(score,view,box)}]
    prompts = args.prompts if args.prompts.rstrip().endswith(".") else args.prompts + "."   # DINO は末尾ピリオド区切り
    for vi, view in enumerate(views):
        img_path = os.path.join(frames_dir, view["name"])   # 視点画像
        if not os.path.isfile(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        dets = detect_and_segment(image, prompts, dino, dino_proc, sam2, sam2_proc,
                                  device, args.box_threshold, args.text_threshold)
        uv, in_front = project_points(means, view)          # ガウシアン投影
        u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)   # 画素インデックス
        inside_img = in_front & (u >= 0) & (u < image.width) & (v >= 0) & (v < image.height)
        print(f"視点 {vi + 1}/{len(views)} ({view['name']}): 検出 {len(dets)} 件")
        for label, score, box, mask in dets:
            member = np.zeros(len(means), dtype=bool)        # このマスクに入ったガウシアン
            idx = np.where(inside_img)[0]
            member[idx] = mask[v[idx], u[idx]]
            if member.sum() < 50:
                continue                                     # 小さすぎる検出は無視
            best_iou, best_obj = 0.0, None                   # 既存物体との対応付け
            for ob in objects:
                inter = np.logical_and(member, ob["member_union"]).sum()   # 交差
                union = np.logical_or(member, ob["member_union"]).sum()    # 和
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou, best_obj = iou, ob
            if best_obj is not None and best_iou > 0.3:      # 同一物体としてマージ
                best_obj["votes"] += member.astype(np.int32)
                best_obj["member_union"] |= member
                best_obj["n_obs"] += 1
                best_obj["label_counts"][label] = best_obj["label_counts"].get(label, 0) + 1
                if score > best_obj["best"][0]:
                    best_obj["best"] = (score, img_path, box)
            else:                                            # 新規物体
                objects.append({"votes": member.astype(np.int32), "member_union": member.copy(),
                                "n_obs": 1, "label_counts": {label: 1},
                                "best": (score, img_path, box)})

    # --- 物体の確定（観測数・投票率・ガウシアン数でフィルタ） ---
    finals = []                                   # 確定した物体
    for ob in objects:
        if ob["n_obs"] < 2:
            continue                              # 1視点のみは誤検出の可能性が高い
        sel = ob["votes"] >= max(2, int(np.ceil(args.min_votes * ob["n_obs"])))   # 投票率で採用
        if sel.sum() < args.min_gaussians:
            continue
        label = max(ob["label_counts"], key=ob["label_counts"].get)   # 多数決ラベル
        finals.append({"sel": sel, "label": label, "score": ob["best"][0],
                       "n_obs": ob["n_obs"], "best": ob["best"]})
    finals.sort(key=lambda o: (o["n_obs"], o["score"]), reverse=True)
    finals = finals[:args.max_objects]            # 上限で切る
    assert finals, ("物体を分離できませんでした。--prompts の英単語を写っている物に合わせる、"
                    "--box-threshold を 0.25 に下げる、を試してください。")

    # --- 物体別の出力（gaussians.ply / mesh.ply / collision / crop.jpg） ---
    from PIL import Image as PILImage             # クロップ保存用
    import coacd                                  # 凸分解
    import trimesh                                # 凸分解結果の保存用
    cfg_mesh = load_recon_config()["mesh"]        # TSDF パラメータの流用
    meta_list = []                                # objects.json 用
    occupied = np.zeros(len(means), dtype=bool)   # いずれかの物体に属したガウシアン
    for oid, ob in enumerate(finals, start=1):
        sel = ob["sel"]                           # この物体のガウシアン
        occupied |= sel
        odir = os.path.join(obj_root, f"obj_{oid}")           # 物体の出力先
        os.makedirs(os.path.join(odir, "collision"), exist_ok=True)
        save_gaussian_ply(os.path.join(odir, "gaussians.ply"), vertex[sel])   # 属性そのまま抽出

        # 物体のみで TSDF → メッシュ（ボクセルは物体サイズ基準で細かく）
        sub = {k: g[k][sel] for k in ("means", "quats", "scales", "opacities", "colors")}   # 部分集合
        bbox_diag = float(np.linalg.norm(sub["means"].max(0) - sub["means"].min(0)) + 1e-9)   # 物体対角
        gt = gaussians_to_torch(sub)              # GPU テンソル化
        mesh_o3d = tsdf_fuse(gt, views, voxel_size=bbox_diag / 128,
                             depth_trunc=bbox_diag * 4, render_scale=0.5, log_every=999)
        import open3d as o3d                      # クラスタ抽出用
        tri_clusters, cluster_n, _ = mesh_o3d.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters); cluster_n = np.asarray(cluster_n)
        if len(cluster_n) > 0:
            mesh_o3d.remove_triangles_by_mask(cluster_n[tri_clusters] < cluster_n.max())   # 最大成分のみ
            mesh_o3d.remove_unreferenced_vertices()
        assert len(mesh_o3d.triangles) > 50, f"obj_{oid} のメッシュ化に失敗（三角形不足）"
        o3d.io.write_triangle_mesh(os.path.join(odir, "mesh.ply"), mesh_o3d)

        # coacd 凸分解（衝突形状）。失敗時は凸包1個にフォールバック
        verts = np.asarray(mesh_o3d.vertices); faces = np.asarray(mesh_o3d.triangles)
        try:
            parts = coacd.run_coacd(coacd.Mesh(verts, faces), threshold=0.07)   # 凸分解
        except Exception as e:
            print(f"obj_{oid}: coacd 失敗（{e}）→ 凸包にフォールバック")
            hull = trimesh.Trimesh(verts, faces, process=False).convex_hull
            parts = [(np.asarray(hull.vertices), np.asarray(hull.faces))]
        for pi, (pv, pf) in enumerate(parts):
            trimesh.Trimesh(np.asarray(pv), np.asarray(pf), process=False).export(
                os.path.join(odir, "collision", f"part_{pi:02d}.obj"))

        # 最良検出のクロップ画像（GPT 物性推定と目視確認に使用）
        score, img_path, box = ob["best"]         # 最良検出
        img = PILImage.open(img_path).convert("RGB")
        x0, y0, x1, y1 = [int(v) for v in box]    # 検出枠
        pad = int(0.1 * max(x1 - x0, y1 - y0))    # 余白
        img.crop((max(0, x0 - pad), max(0, y0 - pad),
                  min(img.width, x1 + pad), min(img.height, y1 + pad))).save(
            os.path.join(odir, "crop.jpg"), quality=90)

        aabb = np.stack([sub["means"].min(0), sub["means"].max(0)])   # COLMAP 単位の AABB
        meta_list.append({"id": oid, "label": ob["label"], "score": round(ob["score"], 3),
                          "n_obs": ob["n_obs"], "n_gaussians": int(sel.sum()),
                          "aabb_colmap": aabb.tolist(), "n_collision_parts": len(parts)})
        print(f"obj_{oid}: {ob['label']}（score {ob['score']:.2f}, 観測 {ob['n_obs']} 視点, "
              f"ガウシアン {sel.sum()}, 凸パーツ {len(parts)}）")

    # --- 静的シーンから物体領域を除去した残余メッシュ ---
    import open3d as o3d                          # 残余メッシュの編集用
    from scipy.spatial import cKDTree             # 近傍検索
    scene_mesh = o3d.io.read_triangle_mesh(os.path.join(ws, "mesh", "scene_mesh.ply"))
    verts = np.asarray(scene_mesh.vertices); tris = np.asarray(scene_mesh.triangles)
    obj_pts = means[occupied]                     # 物体に属するガウシアン中心
    radius = float(np.median(g["scales"][occupied].max(axis=1))) * 3 if occupied.any() else 0.0   # 除去半径
    tree = cKDTree(obj_pts)                       # KD 木
    centroids = verts[tris].mean(axis=1)          # 三角形の重心
    near = tree.query_ball_point(centroids, r=max(radius, 1e-6), return_length=True) > 0   # 物体近傍
    scene_mesh.remove_triangles_by_mask(near)
    scene_mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(os.path.join(obj_root, "static_remainder.ply"), scene_mesh)

    with open(os.path.join(obj_root, "objects.json"), "w", encoding="utf-8") as f:
        json.dump({"objects": meta_list, "prompts": args.prompts}, f, indent=2, ensure_ascii=False)
    save_stats(ws, "objects", {"objects": meta_list, "views_used": len(views),
                               "prompts": args.prompts})
    result_line("OBJECTS", os.path.join(obj_root, "objects.json"))
    result_line("NUM_OBJECTS", len(meta_list))


if __name__ == "__main__":
    main()
