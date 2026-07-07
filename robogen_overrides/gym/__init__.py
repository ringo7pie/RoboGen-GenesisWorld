# ============================================================
# gym → gymnasium 互換 shim
# 役割: RoboGen のコード（と LLM 生成タスクファイル）に大量にある
#       `import gym` / `from gym import spaces` / `from gym.utils import seeding`
#       を、旧 gym をインストールせずに gymnasium で動かすための別名パッケージ。
# 仕組み: このファイルが `import gym` で読み込まれた瞬間に sys.modules['gym'] を
#         gymnasium 本体に差し替える。サブモジュールも同一オブジェクトを共有させることで、
#         isinstance(space, gymnasium.spaces.Box) のような型判定が壊れないようにする。
# 配置: setup_robogen.sh がこのフォルダを RoboGen リポジトリ直下へコピーする
#       （cwd が site-packages より優先されるため、本物の gym が入っていてもこちらが勝つ）。
# ============================================================
import sys                       # sys.modules の書き換え用

import gymnasium                 # 本体（旧 gym の後継パッケージ）
# RoboGen が参照するサブモジュールを先に import しておく（後述のエイリアス登録の対象にするため）
import gymnasium.spaces          # spaces.Box などの空間定義
import gymnasium.utils           # ユーティリティ群
import gymnasium.utils.seeding   # 乱数シード（seeding.np_random）
import gymnasium.envs            # 環境レジストリ
import gymnasium.error           # 例外クラス
import gymnasium.logger          # ロガー

# 'gym' 本体を gymnasium と同一モジュールに差し替える
sys.modules["gym"] = gymnasium

# 'gym.xxx' を 'gymnasium.xxx' と同一オブジェクトとして登録する
# （二重ロードによるクラスの重複定義 = isinstance 判定の破綻を防ぐ）
for _name in [m for m in list(sys.modules) if m.startswith("gymnasium.")]:
    sys.modules["gym" + _name[len("gymnasium"):]] = sys.modules[_name]
