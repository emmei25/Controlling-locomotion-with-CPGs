import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 自作したクラスをインポート
from cpgnet import CPGNetwork

# 出力先ディレクトリの設定（なければ作る）
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# --- 1. Unitree Go1 用のCPG設定 (Trot) ---
intrinsic_freqs = np.ones(4) * 6.0  # 6.0Hz
intrinsic_amps = np.ones(4) * 1.0
phase_biases = np.pi * np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
])
coupling_weights = (phase_biases > 0) * 10.0
convergence_coefs = np.ones(4) * 20.0

network = CPGNetwork(
    timestep=0.002,
    intrinsic_freqs=intrinsic_freqs,
    intrinsic_amps=intrinsic_amps,
    coupling_weights=coupling_weights,
    phase_biases=phase_biases,
    convergence_coefs=convergence_coefs,
)

# --- 2. Simulate the network (元のコードと同じ構成) ---
duration = 2.0  # 2秒間シミュレーション
num_steps = int(duration / network.timestep)

phase_hist = np.empty((num_steps, 4))      # 6足→4足に変更
magnitude_hist = np.empty((num_steps, 4))  # 6足→4足に変更

for i in range(num_steps):
    network.step()
    phase_hist[i, :] = network.curr_phases
    magnitude_hist[i, :] = network.curr_magnitudes

# --- 3. Visualize (元のコードのスタイルを維持) ---
fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
t = np.arange(num_steps) * network.timestep

# 位相のプロット
# 4本の線が描かれます（Trotなら2本ずつ重なって見えるはずです）
axs[0].plot(t, phase_hist % (2 * np.pi), linewidth=1)
axs[0].set_yticks([0, np.pi, 2 * np.pi])
axs[0].set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
axs[0].set_ylabel("Phase")
axs[0].grid(True) # 見やすくするためにGrid追加（任意）

# 振幅のプロット
axs[1].plot(t, magnitude_hist, linewidth=1)
axs[1].set_ylabel("Magnitude")
axs[1].set_xlabel("Time (s)")
axs[1].grid(True)

plt.tight_layout()

# 画像として保存
output_path = output_dir / "go1_cpg_rollout.png"
fig.savefig(output_path)
print(f"グラフを保存しました: {output_path}")

# 画面にも表示する場合
# plt.show()