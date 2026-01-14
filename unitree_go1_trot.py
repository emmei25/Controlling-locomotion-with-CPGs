import numpy as np
from cpgnet import CPGNetwork
from kinematics import get_go1_target_angles
import time
import mujoco
import mujoco.viewer
import imageio  # 動画保存用

def main():
    # --- CPGの設定 (Trot) ---
    intrinsic_freqs = np.ones(4) * 6.0
    intrinsic_amps = np.ones(4) * 1.0
    
    # トロットの位相差行列（対角線を揃える）
    phase_biases = np.pi * np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
    ])
    coupling_weights = (phase_biases > 0) * 10.0
    convergence_coefs = np.ones(4) * 20.0

    # インスタンス化
    cpg = CPGNetwork(
        timestep=0.002,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

# --- 2. MuJoCo読み込み ---
    model_path = "./unitree_go1/scene.xml"
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except ValueError:
        print("エラー: モデルが見つかりません。パスを確認してください。")
        return

    # --- 3. 録画とカメラの設定 ---
    # 動画保存用のレンダラーを作成
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = [] # ここに画像をためていく
    
    # シミュレーション時間とフレームレート
    duration = 5.0      # シミュレーションする時間（秒）
    playback_fps = 30   # 動画ファイルの再生速度（これは30のまま）
    
    # ★スローモーション倍率（ここを好きな数字に変えてください）
    # 2.0 なら 2倍スロー（1/2の速さ）
    # 4.0 なら 4倍スロー（1/4の速さ）
    slow_motion_ratio = 1.0 
    
    total_steps = int(duration / model.opt.timestep)
    
    # 「本来よりたくさん撮る」計算式
    # 1秒間に (30 * 4 = 120枚) 撮ることで、再生時に4倍スローになります
    render_interval = int(1.0 / (playback_fps * slow_motion_ratio) / model.opt.timestep)
    
    # 間隔が0以下にならないよう安全策
    if render_interval < 1:
        render_interval = 1

    print(f"シミュレーション開始 ({duration}秒間)...")

    # Viewerを立ち上げる（画面で見る用）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 1. 本体の高さを調整 (0.35だと高いので 0.29 くらいにします)
        # ※足が地面にちょうどつく高さです
        data.qpos[2] = 0.29
        
        # 2. 関節の初期角度を「立っている状態」にする
        # MuJoCoのqposは [x,y,z, quat(4個), 関節1, 関節2...] の順です
        # したがって、qpos[7] から後ろがモーターの角度です。
        
        # [Hip, Thigh, Calf] の順で、kinematics.pyの基準値と同じにします
        # Thigh=0.8, Calf=-1.6 (前回の修正値を反映)
        standing_pose = [0.0, 0.8, -1.6] 
        
        # 4足分 (12個) のデータを一気に入れます
        # FR, FL, RR, RL すべて同じ姿勢でスタートさせます
        data.qpos[7:19] = np.tile(standing_pose, 4)
        
        # カメラ設定（初期視点）
        viewer.cam.distance = 1.5  # カメラの距離
        viewer.cam.elevation = -20 # カメラの高さ角度
        viewer.cam.azimuth = 90    # 横方向の角度

        # メインループ
        for i in range(total_steps):
            
            # --- A. CPG計算 ---
            cpg.step()
            
            # --- B. マッピング ---
            ctrl_signal = np.zeros(12)
            # FR(0-2) <- CPG[2], FL(3-5) <- CPG[0], RR(6-8) <- CPG[3], RL(9-11) <- CPG[1]
            ctrl_signal[0:3] = get_go1_target_angles(cpg.curr_phases[2], cpg.curr_magnitudes[2])
            ctrl_signal[3:6] = get_go1_target_angles(cpg.curr_phases[0], cpg.curr_magnitudes[0])
            ctrl_signal[6:9] = get_go1_target_angles(cpg.curr_phases[3], cpg.curr_magnitudes[3])
            ctrl_signal[9:12] = get_go1_target_angles(cpg.curr_phases[1], cpg.curr_magnitudes[1])
            
            data.ctrl[:] = ctrl_signal
            
            # --- C. 物理演算 ---
            mujoco.mj_step(model, data)

            # --- D. カメラ追従 (Tracking) ---
            # ロボットの現在位置(x, y, z)を取得
            robot_pos = data.qpos[0:3]
            
            # Viewerのカメラ中心をロボットに合わせる
            viewer.cam.lookat[:] = robot_pos

            # --- E. 画面更新と録画 ---
            if i % render_interval == 0:
                # 1. 画面(Viewer)の更新
                viewer.sync()
                
                # 2. 録画用レンダラーのカメラも更新
                renderer.update_scene(data, camera=viewer.cam)
                
                # 3. 画像を取得してリストに追加
                frame = renderer.render()
                frames.append(frame)

    # --- 4. 動画として保存 ---
    print("動画を保存しています")
    imageio.mimsave("trot.mp4", frames, fps=playback_fps)
    print("完了！")

if __name__ == "__main__":
    main()