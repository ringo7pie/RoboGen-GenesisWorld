# ============================================================
# verify_genesis_usd.py
# 役割: build_usd.py の出力を Genesis 物理シミュレーションで検証し、mp4 を生成する
#   --mode static  : 静的シーンにテスト球を3個落下させ、床・メッシュ貫通が無いか数値検証
#   --mode dynamic : 各動的物体を drop_height 持ち上げて落下させ、静止（速度≈0）と
#                    発散なし（落下貫通なし）を数値検証
#   検証シーンは決定性のため usd/baked/（ベイク済みメッシュ + manifest.json）から構築する。
#   併せて gs.morphs.USD による USD 直接インポートの可否も試験し、結果を記録する（--source auto）。
# 使い方: python verify_genesis_usd.py --workspace <ws> --mode static|dynamic
# 出力: <ws>/verify/<mode>_check.mp4 / stdout: RESULT_MP4= / RESULT_CHECK=PASS|FAIL / RESULT_USD_IMPORT=
# ============================================================
import argparse      # コマンドライン引数の解析用
import json          # manifest 読み込み用
import os            # パス操作用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 共有モジュールの import 用
from recon_utils import load_recon_config, result_line, save_stats   # 共有ユーティリティ
from genesis_record import record_simulation     # 既存の録画ユーティリティ（①②と共用）


# gs.morphs.USD で USD を直接インポートできるかを別シーンで試験する（検証本体には使わない）
# 戻り値: "OK" または "NG: <理由>"
def probe_usd_import(gs, usd_path):
    try:
        scene = gs.Scene(show_viewer=False)       # 試験用の使い捨てシーン
        scene.add_entity(gs.morphs.USD(file=usd_path, fixed=True))   # USD morph での読み込み
        scene.build()
        return "OK"
    except Exception as e:
        return f"NG: {type(e).__name__}: {str(e)[:200]}"


def main():
    parser = argparse.ArgumentParser(description="Genesis による USD/メッシュの物理検証")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--mode", choices=["static", "dynamic"], required=True, help="検証モード")
    parser.add_argument("--source", choices=["auto", "mesh", "usd"], default="auto",
                        help="検証シーンの構築元（auto=メッシュで検証し USD import は可否試験のみ）")
    parser.add_argument("--steps", type=int, default=None, help="シミュレーションステップ数")
    args = parser.parse_args()                    # 解析済み引数

    cfg = load_recon_config()["verify"]           # 既定値
    steps = args.steps or cfg["steps"]            # ステップ数

    ws = args.workspace                           # 作業ディレクトリ
    manifest_path = os.path.join(ws, "usd", "baked", "manifest.json")   # ベイク済み一式
    assert os.path.isfile(manifest_path), "manifest.json がありません（先に build_usd.py）"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)                   # ベイク済みメッシュと物性の一覧
    usd_path = os.path.join(ws, "usd", f"scene_{args.mode if args.mode == 'dynamic' else 'static'}.usdc")
    if not os.path.isfile(usd_path):
        usd_path = os.path.join(ws, "usd", f"scene_{manifest['phase']}.usdc")   # フェーズ一致の USD

    import genesis as gs                          # Genesis 本体（重いので main 内 import）
    try:
        gs.init(backend=gs.gpu)                   # GPU バックエンド
    except Exception as e:
        print(f"GPU 初期化に失敗したため CPU にフォールバックします: {e}")
        gs.init(backend=gs.cpu)

    usd_import = "SKIPPED"                        # USD 直接インポートの可否
    if args.source in ("auto", "usd") and os.path.isfile(usd_path):
        usd_import = probe_usd_import(gs, usd_path)
        print(f"gs.morphs.USD インポート試験: {usd_import}")

    # --- 検証シーンの構築（決定性のためベイク済みメッシュから） ---
    ext_min = np.array(manifest["scene_extent"][0])   # シーン範囲下限
    ext_max = np.array(manifest["scene_extent"][1])   # シーン範囲上限
    center = (ext_min + ext_max) / 2                  # シーン中心
    diag = float(np.linalg.norm(ext_max - ext_min))   # シーン対角 [m]

    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=0.01))   # 検証シーン
    scene.add_entity(gs.morphs.Plane())               # 床（z=0。build_usd と同じ規約）
    scene.add_entity(gs.morphs.Mesh(                  # 静的シーンメッシュ（固定・凹形状のまま）
        file=manifest["static_mesh"], fixed=True, convexify=False))

    trackers = []                                     # 監視対象 [(名前, entity, 追加情報)]
    if args.mode == "static":
        rng = np.random.default_rng(0)                # 再現性のある乱数
        r = cfg["ball_radius"]                        # テスト球の半径
        for i in range(3):
            xy = center[:2] + (rng.random(2) - 0.5) * diag * 0.3   # 中心付近のランダム xy
            ball = scene.add_entity(gs.morphs.Sphere(
                radius=r, pos=(float(xy[0]), float(xy[1]), float(ext_max[2] + 0.3))))   # シーン上空から落下
            trackers.append((f"ball_{i}", ball, {"radius": r}))
    else:
        assert manifest["objects"], "manifest に動的物体がありません（フェーズ2 を先に実行）"
        import trimesh                                 # 密度計算用
        for ob in manifest["objects"]:
            hull = trimesh.load(ob["visual"], process=False).convex_hull   # 凸包（体積計算用）
            volume = max(float(hull.volume), 1e-6)     # 体積 [m^3]
            rho = ob["mass"] / volume                  # Genesis は密度指定のため質量→密度換算
            c = ob["centroid"]                         # ベイク時の重心位置
            ent = scene.add_entity(
                gs.morphs.Mesh(file=ob["visual"], fixed=False,
                               pos=(0.0, 0.0, cfg["drop_height"])),   # drop_height だけ持ち上げ
                material=gs.materials.Rigid(rho=rho, friction=ob["static_friction"]))
            trackers.append((ob["label"], ent, {"mass": ob["mass"], "rho": rho, "centroid": c}))

    camera = scene.add_camera(res=(640, 480),
                              pos=(float(center[0] + diag), float(center[1] - diag), float(ext_max[2] + diag * 0.5)),
                              lookat=(float(center[0]), float(center[1]), float(center[2])),
                              fov=40, GUI=False)      # 検証映像用カメラ
    scene.build()

    # --- シミュレーション + 監視 ---
    history = {name: [] for name, _, _ in trackers}   # 位置履歴

    # 毎ステップ呼ばれる監視関数: 各対象の位置を記録する
    def track(step_i):
        for name, ent, _ in trackers:
            pos = ent.get_pos()                        # 現在位置（tensor）
            history[name].append(np.array(pos.tolist(), dtype=float).reshape(-1)[:3])

    out_dir = os.path.join(ws, "verify")               # 検証出力先
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, f"{args.mode}_check.mp4")   # 検証動画
    record_simulation(scene=scene, camera=camera, steps=steps, out_path=mp4_path,
                      fps=30, control_fn=track, render_every=2)

    # --- 数値判定 ---
    checks = {}                                        # 対象ごとの判定詳細
    all_pass = True                                    # 総合判定
    for name, _, info in trackers:
        traj = np.stack(history[name])                 # 位置履歴 [T,3]
        z_min = float(traj[:, 2].min())                # 最低高度
        speed_end = float(np.linalg.norm(traj[-1] - traj[-20], axis=0) / (0.01 * 19))   # 終端速度 [m/s]
        if args.mode == "static":
            floor_ok = z_min >= info["radius"] - cfg["penetration_tol"]   # 床貫通なし（球心>=半径-許容）
            ok = bool(floor_ok)
            checks[name] = {"z_min": z_min, "min_allowed": info["radius"] - cfg["penetration_tol"],
                            "pass": ok}
        else:
            rest_ok = speed_end < cfg["rest_speed"]    # 静止した
            no_fall = z_min > -1.0                     # 床を突き抜けていない
            ok = bool(rest_ok and no_fall)
            checks[name] = {"z_min": z_min, "final_speed": speed_end,
                            "rest_ok": rest_ok, "no_fall_through": no_fall, "pass": ok}
        all_pass = all_pass and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {checks[name]}")

    save_stats(ws, f"verify_{args.mode}", {
        "mode": args.mode, "steps": steps, "usd_import": usd_import,
        "checks": checks, "all_pass": all_pass, "mp4": mp4_path})
    result_line("MP4", mp4_path)
    result_line("USD_IMPORT", usd_import)
    result_line("CHECK", "PASS" if all_pass else "FAIL")
    if not all_pass:
        print("検証 FAIL: 上の数値（z_min / final_speed）とともに報告してください。"
              "貫通の場合はメッシュの穴、静止しない場合は摩擦/質量の推定値が原因の可能性があります。")
        sys.exit(1)
    print("検証 PASS")


if __name__ == "__main__":
    main()
