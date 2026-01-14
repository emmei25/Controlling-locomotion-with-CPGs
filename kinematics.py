import numpy as np

def get_go1_target_angles(phase, amplitude):
    """
    CPGの位相(0~2π)を受け取り、Go1の片足3関節の角度を計算する
    戻り値: [Hip, Thigh, Calf] の角度
    """
    s = np.sin(phase)

    # 1. Hip (股関節・横)
    angle_hip = 0.0

    # 2. Thigh (太もも)
    # 0.9 rad を基準に振る
    angle_thigh = 0.8 - (amplitude * 0.3) * s

    # 3. Calf (膝)
    base_calf = -1.6
    if s > 0:
        # 足上げ期
        angle_calf = base_calf - (amplitude * 0.7) * s
    else:
        # 接地期
        angle_calf = base_calf

    return np.array([angle_hip, angle_thigh, angle_calf])