import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):

    # 坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if abs(L) < 1e-12:
        raise ValueError("错误：两个节点坐标重合，单元无效！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 构造6×6刚度矩阵
    EA_L = E * A / L
    Ke = np.zeros((6, 6))

    # 按公式赋值
    Ke[0, 0] = cx**2
    Ke[0, 1] = cx * cy
    Ke[0, 2] = cx * cz
    Ke[0, 3] = -cx**2
    Ke[0, 4] = -cx * cy
    Ke[0, 5] = -cx * cz

    Ke[1, 0] = cx * cy
    Ke[1, 1] = cy**2
    Ke[1, 2] = cy * cz
    Ke[1, 3] = -cx * cy
    Ke[1, 4] = -cy**2
    Ke[1, 5] = -cy * cz

    Ke[2, 0] = cx * cz
    Ke[2, 1] = cy * cz
    Ke[2, 2] = cz**2
    Ke[2, 3] = -cx * cz
    Ke[2, 4] = -cy * cz
    Ke[2, 5] = -cz**2

    Ke[3, 0] = -cx**2
    Ke[3, 1] = -cx * cy
    Ke[3, 2] = -cx * cz
    Ke[3, 3] = cx**2
    Ke[3, 4] = cx * cy
    Ke[3, 5] = cx * cz

    Ke[4, 0] = -cx * cy
    Ke[4, 1] = -cy**2
    Ke[4, 2] = -cy * cz
    Ke[4, 3] = cx * cy
    Ke[4, 4] = cy**2
    Ke[4, 5] = cy * cz

    Ke[5, 0] = -cx * cz
    Ke[5, 1] = -cy * cz
    Ke[5, 2] = -cz**2
    Ke[5, 3] = cx * cz
    Ke[5, 4] = cy * cz
    Ke[5, 5] = cz**2

    Ke = EA_L * Ke
    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元应变、应力、轴力
    :param x1: 节点1坐标
    :param x2: 节点2坐标
    :param E: 弹性模量
    :param A: 横截面积
    :param de: 节点位移列阵 [u1,v1,w1,u2,v2,w2]
    :return: epsilon, sigma, N
    """
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    if abs(L) < 1e-12:
        raise ValueError("错误：两个节点坐标重合，单元无效！")

    cx = dx / L
    cy = dy / L
    cz = dz / L

    # 应变位移矩阵 B
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = np.dot(B, de)      # 应变
    sigma = E * epsilon          # 应力 (Pa)
    N = sigma * A                # 轴力 (N)
    return epsilon, sigma, N


def main():
    print("===== 三维杆单元算例验证 =====\n")

    # ========== 算例1：沿X轴一维杆单元 ==========
    print("---------- 算例1：沿X轴一维杆单元 ----------")
    x1_1 = [0, 0, 0]
    x2_1 = [2, 0, 0]
    E1 = 200e9       # 200 GPa
    A1 = 1.0e-4      # m²
    de1 = np.array([0, 0, 0, 1.0e-3, 0, 0])

    L1, cos1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
    eps1, sig1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

    print(f"单元长度 L = {L1:.4f} m")
    print(f"方向余弦 cx,cy,cz = {cos1[0]:.4f}, {cos1[1]:.4f}, {cos1[2]:.4f}")
    print("全局刚度矩阵 Ke：")
    print(np.round(Ke1, 2))
    print(f"轴向应变 ε = {eps1:.6e}")
    print(f"轴向应力 σ = {sig1/1e6:.2f} MPa")
    print(f"轴力 N = {N1:.2f} N\n")

    # ========== 算例2：空间任意方向杆单元 ==========
    print("---------- 算例2：空间任意方向杆单元 ----------")
    x1_2 = [0, 0, 0]
    x2_2 = [1, 2, 2]
    E2 = 210e9       # 210 GPa
    A2 = 2.0e-4      # m²
    de2 = np.array([0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])

    L2, cos2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
    eps2, sig2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

    print(f"单元长度 L = {L2:.4f} m")
    print(f"方向余弦 cx,cy,cz = {cos2[0]:.4f}, {cos2[1]:.4f}, {cos2[2]:.4f}")
    # 验证刚度矩阵对称性
    is_symmetric = np.allclose(Ke2, Ke2.T)
    print(f"刚度矩阵是否对称：{is_symmetric}")
    print(f"轴向应变 ε = {eps2:.6e}")
    print(f"轴向应力 σ = {sig2/1e6:.2f} MPa")
    print(f"轴力 N = {N2:.2f} N\n")

    # ========== 刚体位移验证 ==========
    print("---------- 刚体平移位移验证 ----------")
    # 整体平移：所有节点位移相同，无变形、无内力
    de_rigid = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    eps_r, sig_r, N_r = truss3d_element_stress(x1_2, x2_2, E2, A2, de_rigid)
    print(f"刚体平移下应变 ε = {eps_r:.2e}")
    print(f"刚体平移下轴力 N = {N_r:.2e}\n")

    # ========== 刚度矩阵物理意义验证 ==========
    print("---------- 刚度矩阵物理意义验证 ----------")
    # 令第4个自由度位移=1，其余为0
    de_test = np.zeros(6)
    de_test[3] = 1.0
    Fe = np.dot(Ke2, de_test)
    print("指定第4自由度单位位移，节点力列阵 Fe：")
    print(np.round(Fe, 2))
    print("结论：Fe 与刚度矩阵 Ke 第4列完全一致")


if __name__ == "__main__":
    main()