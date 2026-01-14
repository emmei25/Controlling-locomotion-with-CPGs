import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ★ 計算式（kinematics）をインポート
from kinematics import get_go1_target_angles

# 出力先
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# --- 1. データの準備 ---

# Go1の関節定義
legs = [f"{pos}{side}" for side in "LR" for pos in "FR"] # FL, RL, FR, RR
dofs_per_leg = [
    "Hip",
    "Thigh",
    "Calf",
]

# 3サイクル分 (3 * 2π) の時間軸
theta_ts = np.linspace(0, 3 * 2 * np.pi, 1000)

# ★振幅の変化 (0から1へ徐々に増加)
r_ts = np.linspace(0, 1, 1000)

# 位相 -> 角度 の関数化 (psi_funcs)
# kinematics.py の関数は (phase, amplitude) を受け取りますが、
# ここでは amplitude=1.0 (最大振幅) の波形を基準波形として定義します
psi_funcs = {}
for leg in legs:
    # 常に最大振幅の波形を返す関数として定義
    def make_func(l_name):
        return lambda t_array: np.array([get_go1_target_angles(t, amplitude=1.0) for t in t_array]).T
    psi_funcs[leg] = make_func(leg)

# ##### THIS SECTION HAS CHANGED (振幅変調の適用) #####
joint_angles_by_leg = {}
for leg, psi_func in psi_funcs.items():
    # phase=0 のときの姿勢を「中立姿勢（Neutral Position）」とみなします
    neutral_pos = psi_func([0])
    
    # 基本波形を計算
    base_wave = psi_func(theta_ts) # (3, 1000)
    
    # 振幅変調の計算式: 
    # Current = Neutral + r * (Target_at_Max_Amp - Neutral)
    # rが0ならNeutralのまま、rが1なら最大振幅の波形になります
    joint_angles_by_leg[leg] = neutral_pos + r_ts * (base_wave - neutral_pos)
#####################################################

# --- 2. 描画 (Visualize) ---
# Unitree Go1用に (3x2) -> (2x2) に変更しつつ、スタイルを踏襲

fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)

for i_side, side in enumerate("LR"):     # 0:L, 1:R
    for i_pos, pos in enumerate("FR"):   # 0:F, 1:R
        
        leg = f"{pos}{side}" # FL, FR, RL, RR
        ax = axs[i_pos, i_side]
        
        joint_angles = np.rad2deg(joint_angles_by_leg[leg])
        
        for i_dof, dof_name in enumerate(dofs_per_leg):
            legend = dof_name if i_pos == 0 and i_side == 0 else None
            ax.plot(theta_ts, joint_angles[i_dof, :], linewidth=1, label=legend)

        # X軸の設定 (一番下の行だけ)
        if i_pos == 1:
            ax.set_xlabel("Phase")
            ax.set_xticks(np.pi * np.arange(7))
            ax.set_xticklabels(["0" if x == 0 else rf"{x}$\pi$" for x in np.arange(7)])
        
        # Y軸の設定 (一番左の列だけ)
        if i_side == 0:
            ax.set_ylabel(r"DoF angle ($\degree$)")
        
        ax.set_title(f"{leg} leg")
        
        # 可動範囲の設定
        ax.set_ylim(-100, 100)
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.grid(True, linestyle=":", alpha=0.6)

fig.legend(loc="center right")
fig.tight_layout()
fig.subplots_adjust(right=0.8)
fig.savefig(output_dir / "go1_three_steps_amp_modulated.png")

print(f"グラフを保存しました: {output_dir / 'go1_three_steps_amp_modulated.png'}")
# plt.show()