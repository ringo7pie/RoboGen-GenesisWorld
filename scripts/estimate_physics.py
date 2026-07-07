# ============================================================
# estimate_physics.py
# 役割: 分離した各物体の物理パラメータを GPT-4o で推定する（NeRF2Physics 方式）
#   ① 物体のクロップ画像+ラベルを GPT-4o に送り、材質・密度・摩擦・反発を JSON で取得
#   ② configs/materials_db.yaml の材質別レンジで clamp（物理的に妥当な範囲へ）
#   ③ 質量 = 密度 × 実寸体積（メッシュ体積 × scale^3。非 watertight は凸包×0.7 で近似）
# 使い方: python estimate_physics.py --workspace <ws>
# 出力: <ws>/physics/physics.json（手修正可。修正後は build_usd.py から再実行すればよい）
# stdout: RESULT_PHYSICS=<path>
# ============================================================
import argparse      # コマンドライン引数の解析用
import base64        # 画像送信用
import json          # 応答・出力の JSON 処理用
import os            # パス操作用
import sys           # 終了コード用

import numpy as np   # 数値計算用

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # recon_utils の import 用
from recon_utils import load_scale, result_line, save_stats      # 共有ユーティリティ

# GPT への指示（材質と物性を JSON で返させる）
SYSTEM_PROMPT = """あなたは家庭・オフィスにある物体の物理特性を推定する専門家です。
与えられた物体の写真とラベルから、以下を JSON で返してください:
- material: 主要材質。次のいずれか: wood, metal, plastic, glass, ceramic, fabric,
  paper, rubber, stone, concrete, leather, foam, cardboard, composite, unknown
- density_kg_m3: 物体全体の実効密度 [kg/m^3]（中空構造なら見かけ密度。例: 空のペットボトル≈50）
- static_friction / dynamic_friction: 一般的な床材（木・タイル）に対する摩擦係数
- restitution: 反発係数（0=弾まない, 1=完全弾性）
- movable: 人が手で動かせる物体か（家具の作り付け・大型家電は false）
- confidence: 推定の自信 (0-1)"""

RESPONSE_SCHEMA = {   # structured output 用スキーマ
    "type": "object",
    "properties": {
        "material": {"type": "string"},
        "density_kg_m3": {"type": "number"},
        "static_friction": {"type": "number"},
        "dynamic_friction": {"type": "number"},
        "restitution": {"type": "number"},
        "movable": {"type": "boolean"},
        "confidence": {"type": "number"}},
    "required": ["material", "density_kg_m3", "static_friction", "dynamic_friction",
                 "restitution", "movable", "confidence"],
    "additionalProperties": False}


# 値を [lo, hi] に clamp し、範囲外だったかを返す
def clamp(value, lo, hi):
    c = float(min(max(value, lo), hi))            # clamp 後の値
    return c, (c != value)


# 物体メッシュの実寸体積 [m^3] を求める（watertight でなければ凸包×0.7 で近似）
def baked_volume(mesh_path, scale):
    import trimesh                                # 体積計算用
    mesh = trimesh.load(mesh_path, process=False) # 物体メッシュ（COLMAP 単位）
    if mesh.is_watertight and mesh.volume > 0:
        vol_colmap = float(mesh.volume)           # そのままの体積
        method = "watertight"
    else:
        vol_colmap = float(mesh.convex_hull.volume) * 0.7   # 凸包の 7 割で近似
        method = "convex_hull*0.7"
    return vol_colmap * scale ** 3, method        # 実寸化（係数の3乗）


def main():
    parser = argparse.ArgumentParser(description="GPT-4o による物体物性の推定")
    parser.add_argument("--workspace", required=True, help="作業ディレクトリ")
    parser.add_argument("--model", default=os.environ.get("RECON_LLM_MODEL", "gpt-4o"), help="使用モデル")
    parser.add_argument("--materials-db", default=None, help="材質レンジ表の YAML パス")
    args = parser.parse_args()                    # 解析済み引数

    import yaml                                   # materials_db の読込用
    from openai import OpenAI                     # GPT 呼び出し用
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY が未設定です（セル18 を先に実行）"
    client = OpenAI()                             # API クライアント

    db_path = args.materials_db or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "configs", "materials_db.yaml")   # レンジ表
    with open(db_path, encoding="utf-8") as f:
        db = yaml.safe_load(f)                    # 材質→レンジ
    mass_lo, mass_hi = db.pop("mass_range")       # 質量のサニティ範囲

    ws = args.workspace                           # 作業ディレクトリ
    with open(os.path.join(ws, "objects", "objects.json"), encoding="utf-8") as f:
        objects = json.load(f)["objects"]         # 物体一覧
    scale_info = load_scale(ws)                   # 実寸換算係数
    scale = float(scale_info["scale_factor"])
    if scale_info["confidence"] == "low":
        print("警告: スケール校正の信頼度が low です（質量は係数の3乗で効くため誤差が拡大します）")

    results = {}                                  # id → 物性
    rows = []                                     # 表示用テーブル
    for ob in objects:
        oid = str(ob["id"])                       # 物体 ID
        crop = os.path.join(ws, "objects", f"obj_{oid}", "crop.jpg")   # クロップ画像
        mesh = os.path.join(ws, "objects", f"obj_{oid}", "mesh.ply")   # 物体メッシュ
        with open(crop, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()   # 画像を base64 化
        res = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": [
                          {"type": "text", "text": f"物体ラベル: {ob['label']}"},
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "physics", "strict": True, "schema": RESPONSE_SCHEMA}},
            temperature=0)
        ph = json.loads(res.choices[0].message.content)   # GPT の推定値

        mat = ph["material"] if ph["material"] in db else "unknown"   # 未知材質は unknown 扱い
        rng = db[mat]                              # この材質の許容レンジ
        clamped = []                               # clamp が発動した項目
        for key, rkey in (("density_kg_m3", "density"), ("static_friction", "static_friction"),
                          ("dynamic_friction", "dynamic_friction"), ("restitution", "restitution")):
            ph[key], hit = clamp(ph[key], rng[rkey][0], rng[rkey][1])
            if hit:
                clamped.append(key)

        volume, vol_method = baked_volume(mesh, scale)    # 実寸体積
        mass = ph["density_kg_m3"] * volume               # 質量 = 密度×体積
        mass, mass_hit = clamp(mass, mass_lo, mass_hi)    # サニティ範囲で clamp
        if mass_hit:
            clamped.append("mass")

        results[oid] = {"material": mat, "density_kg_m3": ph["density_kg_m3"],
                        "static_friction": ph["static_friction"],
                        "dynamic_friction": ph["dynamic_friction"],
                        "restitution": ph["restitution"], "movable": ph["movable"],
                        "confidence": ph["confidence"], "mass": round(mass, 4),
                        "volume_m3": round(volume, 6), "volume_method": vol_method,
                        "clamped": clamped}
        rows.append((oid, ob["label"], mat, ph["density_kg_m3"], round(mass, 3),
                     ph["static_friction"], ph["restitution"], ph["movable"], ",".join(clamped) or "-"))

    # --- 表の表示と保存 ---
    print("\n id | label            | 材質      | 密度   | 質量kg | 静摩擦 | 反発 | 可動 | clamp")
    print("-" * 90)
    for r in rows:
        print(f" {r[0]:>2} | {r[1][:16]:<16} | {r[2]:<9} | {r[3]:>6.0f} | {r[4]:>6} | {r[5]:>6.2f} | {r[6]:>4.2f} | {r[7]!s:<5} | {r[8]}")
    out_path = os.path.join(ws, "physics", "physics.json")   # 出力先
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"objects": results, "model": args.model,
                   "scale_confidence": scale_info["confidence"]}, f, indent=2, ensure_ascii=False)
    print(f"\n物性を保存しました: {out_path}")
    print("値を手修正したい場合はこの JSON を編集し、build_usd.py（セル21）から再実行してください。")
    save_stats(ws, "physics", {"objects": results, "model": args.model})
    result_line("PHYSICS", out_path)


if __name__ == "__main__":
    main()
