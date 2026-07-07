# ============================================================
# genesis_record.py
# 役割: Genesis のシーンをヘッドレスでシミュレーションしながらカメラ録画し、
#       mp4 として保存する共通ユーティリティ（ノートブック①②で共用）
# 前提: gs.init() 済み、scene.build() 済みであること
# ============================================================
import os        # 出力ディレクトリ作成・ファイルサイズ確認用


# シーンを steps 回ステップ実行しながら録画し、mp4 に保存する
# 引数:
#   scene      : 構築済み (build 済み) の gs.Scene
#   camera     : scene.add_camera() で追加したカメラ
#   steps      : シミュレーションのステップ数
#   out_path   : 出力する mp4 のパス
#   fps        : 動画のフレームレート
#   control_fn : 毎ステップ呼ばれるコールバック control_fn(step_index)。関節駆動などに使う（省略可）
#   render_every: 何ステップごとに1フレーム録画するか（描画コスト削減用）
# 戻り値: (保存したファイルパス, 録画フレーム数)
def record_simulation(scene, camera, steps, out_path,
                      fps=30, control_fn=None, render_every=2):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  # 出力先ディレクトリを確保

    camera.start_recording()                     # カメラ録画の開始
    frames = 0                                   # 録画したフレーム数
    for i in range(steps):
        if control_fn is not None:
            control_fn(i)                        # 関節目標の更新などユーザー定義の制御
        scene.step()                             # 物理シミュレーションを1ステップ進める
        if i % render_every == 0:
            camera.render()                      # 現在のフレームを録画バッファに追加
            frames += 1
    camera.stop_recording(save_to_filename=out_path, fps=fps)  # mp4 へ書き出し

    # --- 保存結果の検証（サイズ0の動画を「成功」と誤認しないため） ---
    size = os.path.getsize(out_path)             # 出力ファイルのバイトサイズ
    assert size > 0, f"動画ファイルが空です: {out_path}"
    print(f"録画完了: {out_path} ({size / 1024:.1f} KB, {frames} フレーム, {fps} fps)")
    return out_path, frames
