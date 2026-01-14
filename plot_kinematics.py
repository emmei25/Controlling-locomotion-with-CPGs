import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ★ 計算式（kinematics）をインポート
# データファイルの代わりになります
from kinematics import get_go1_target_angles

# 出力先
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# --- 1. データの準備 (Load & Interpolate の代わり) ---

# Go1の関節定義
legs = [f"{pos}{side}" for side in "LR" for pos in "FR"] # FL, RL, FR, RR
dofs_per_leg = [
    "Hip",
    "Thigh",
    "Calf",
]

# 3サイクル分 (3 * 2π) の時間軸を作成
theta_ts = np.linspace(0, 3 * 2 * np.pi, 1000)

# psi_funcs (位相 -> 角度 の関数) の辞書を作成
# 元のコードでは CubicSpline ですが、ここでは計算式をラップして同じように振る舞わせます
psi_funcs = {}

for leg in legs:
    # ラムダ式を使って、配列 t を受け取って角度配列を返す関数を作る
    # get_go1_target_angles はスカラ入力用なので、配列対応させる
    def make_func(l_name):
        return lambda t_array: np.array([get_go1_target_angles(t, amplitude=1.0) for t in t_array]).T
    
    psi_funcs[leg] = make_func(leg)

# 各足の角度データを計算
joint_angles_by_leg = {}
for leg, psi_func in psi_funcs.items():
    joint_angles_by_leg[leg] = psi_func(theta_ts)

# --- 2. 描画 (Visualize) ---
# 元のコードの書き方を踏襲
# Flyは6本足(3x2)でしたが、Go1は4本足なので(2x2)にします

fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)

# "LR" (左右) と "FR" (前後) でループを回すスタイルを維持
for i_side, side in enumerate("LR"):     # 0:L, 1:R
    for i_pos, pos in enumerate("FR"):   # 0:F, 1:R (Rear)
        
        # Go1の命名規則 (FL, FR, RL, RR) に合わせる
        # 元コードは side+pos でした
        leg = f"{pos}{side}" 
        
        # サブプロットの選択
        ax = axs[i_pos, i_side]
        
        # 角度データの取得と度数法への変換
        joint_angles = np.rad2deg(joint_angles_by_leg[leg])
        
        for i_dof, dof_name in enumerate(dofs_per_leg):
            # 凡例は左上(0,0)だけに表示
            legend = dof_name if i_pos == 0 and i_side == 0 else None
            
            # プロット
            ax.plot(theta_ts, joint_angles[i_dof, :], linewidth=1, label=legend)

        # X軸の設定 (一番下の行だけ)
        if i_pos == 1: 
            ax.set_xlabel("Phase")
            ax.set_xticks(np.pi * np.arange(7))
            ax.set_xticklabels(["0" if x == 0 else rf"{x}$\pi$" for x in np.arange(7)])
        
        # Y軸の設定 (一番左の列だけ)
        if i_side == 0:
            ax.set_ylabel(r"DoF angle ($\degree$)")
        
        # タイトル
        ax.set_title(f"{leg} leg")
        
        # 範囲設定
        # Go1の可動域に合わせて調整 (-180~180だとスカスカになるため狭めています)
        ax.set_ylim(-100, 100)
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.grid(True, linestyle=":", alpha=0.6)

# 凡例とレイアウト調整
fig.legend(loc="center right") # 位置調整
fig.tight_layout()
fig.subplots_adjust(right=0.8) # 凡例スペース確保

# 保存
fig.savefig(output_dir / "go1_three_steps_phase_only.png")
print(f"グラフを保存しました: {output_dir / 'go1_three_steps_phase_only.png'}")

# plt.show()