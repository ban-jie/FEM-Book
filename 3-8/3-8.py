import numpy as np
import matplotlib.pyplot as plt
import warnings

# ==================== 全局绘图&警告配置 ====================
# 屏蔽全部字体警告
warnings.filterwarnings("ignore")
# 中文、负号、Windows绘图后端兼容
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["backend"] = "TkAgg"

# ==================== 通用基础工具函数 ====================
def coth(x):
    """双曲余切，用于SUPG稳定参数计算"""
    return np.cosh(x) / np.sinh(x)

def alpha_supg(Pe):
    """SUPG最优稳定参数 alpha_opt = coth(Pe) - 1/Pe"""
    if abs(Pe) < 1e-8:
        return 0.0
    return coth(Pe) - 1.0 / Pe

def element_matrix(kappa, v, le, alpha):
    """生成2节点线性单元对流扩散刚度矩阵"""
    kappa_bar = kappa + alpha * v * le / 2.0
    K_diff = (kappa_bar / le) * np.array([[1, -1], [-1, 1]])
    K_adv = (v / 2.0) * np.array([[-1, 1], [-1, 1]])
    return K_diff + K_adv

def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    有限元求解主函数
    返回：节点坐标x、数值解theta_num、解析解theta_exact、全局刚度矩阵K
    """
    le = L / nel
    nnodes = nel + 1
    x = np.linspace(0, L, nnodes)
    K = np.zeros((nnodes, nnodes))
    b = np.zeros(nnodes)

    # 全局矩阵组装
    for e in range(nel):
        node_ids = [e, e + 1]
        Ke = element_matrix(kappa, v, le, alpha)
        for i in range(2):
            for j in range(2):
                K[node_ids[i], node_ids[j]] += Ke[i, j]

    # 施加本质边界条件 θ(0)=0, θ(L)=1
    K[0, :] = 0.0
    K[0, 0] = 1.0
    b[0] = 0.0
    K[-1, :] = 0.0
    K[-1, -1] = 1.0
    b[-1] = 1.0

    theta_num = np.linalg.solve(K, b)
    # 稳定解析解，避免大数溢出
    arg = v / kappa
    theta_exact = np.expm1(arg * x) / np.expm1(arg * L)
    return x, theta_num, theta_exact, K

def check_matrix_property(K):
    """检验全局刚度矩阵对称性、正定性"""
    is_symmetric = np.allclose(K, K.T, atol=1e-10)
    eigvals = np.linalg.eigvals(K)
    min_eig = np.min(np.real(eigvals))
    is_positive_definite = min_eig > -1e-10
    return is_symmetric, is_positive_definite, min_eig

# ==================== 基础作业主程序（Pe=0.1、Pe=3，nel=20） ====================
def base_case_solver():
    L = 1.0
    nel = 20
    v = 1.0
    le = L / nel
    Pe_list = [0.1, 3.0]
    error_table = []

    for Pe in Pe_list:
        print(f"========== 当前Pe = {Pe} ==========")
        kappa = (v * le) / (2 * Pe)
        print(f"扩散系数 kappa = {kappa:.6f}")

        alpha_gal = 0.0
        alpha_upwind = 1.0
        alpha_sup = alpha_supg(Pe)
        print(f"SUPG最优alpha = {alpha_sup:.6f}")

        # 三种格式求解
        x_gal, theta_gal, theta_ex, K_gal = solve_advection_diffusion(nel, L, v, kappa, alpha_gal)
        x_up, theta_up, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha_upwind)
        x_sup, theta_sup, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha_sup)

        # 计算最大节点误差
        err_gal = np.max(np.abs(theta_gal - theta_ex))
        err_up = np.max(np.abs(theta_up - theta_ex))
        err_sup = np.max(np.abs(theta_sup - theta_ex))
        error_table.append([Pe, "Galerkin", err_gal])
        error_table.append([Pe, "迎风", err_up])
        error_table.append([Pe, "SUPG", err_sup])
        print(f"Galerkin最大误差: {err_gal:.6e}")
        print(f"迎风格式最大误差: {err_up:.6e}")
        print(f"SUPG最大误差: {err_sup:.6e}")

        # Pe=3时矩阵性质分析
        if abs(Pe - 3.0) < 1e-6:
            sym, pd, mineig = check_matrix_property(K_gal)
            print("\n==== Pe=3 Galerkin矩阵性质 ====")
            print(f"矩阵对称：{sym}，最小特征值：{mineig:.6e}，正定：{pd}")
            print("结论：对流项破坏矩阵对称与正定，高Pe产生数值振荡\n")

        # 绘制单Pe对比曲线
        plt.figure(figsize=(10, 6))
        plt.plot(x_gal, theta_ex, "k-", lw=2, label="精确解")
        plt.plot(x_gal, theta_gal, "b--", lw=1.5, label="标准Galerkin(α=0)")
        plt.plot(x_up, theta_up, "r-.", lw=1.5, label="迎风差分(α=1)")
        plt.plot(x_sup, theta_sup, "g-", lw=1.5, label="SUPG(α_opt)")
        plt.xlabel("x")
        plt.ylabel("θ(x)")
        plt.title(f"一维对流扩散解对比 Pe={Pe}, nel={nel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show(block=True)
        plt.close()

    # 输出总误差表
    print("==================== 基础工况误差汇总 ====================")
    print(f"{'Peclet数':<10}{'格式':<12}{'最大节点误差':<16}")
    for pe, name, err in error_table:
        print(f"{pe:<10.1f}{name:<12}{err:<16.6e}")

# ==================== 附加题：网格加密收敛分析（nel=10/20/40/80，固定Pe=3） ====================
def grid_convergence_study():
    L = 1.0
    v = 1.0
    Pe_target = 3.0
    nel_list = [10, 20, 40, 80]
    err_gal, err_up, err_sup = [], [], []

    print("\n===== 附加题：网格加密误差表（固定Pe=3）=====")
    print(f"{'单元数':<6}{'Galerkin误差':<14}{'迎风误差':<14}{'SUPG误差':<14}")
    for nel in nel_list:
        le = L / nel
        # 每一组网格独立计算kappa，保证Pe恒等于3（修复误差不变bug）
        kappa = (v * le) / (2 * Pe_target)

        _, t_gal, tex, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=0)
        eg = np.max(np.abs(t_gal - tex))
        _, t_up, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=1)
        eu = np.max(np.abs(t_up - tex))
        a_opt = alpha_supg(Pe_target)
        _, t_sup, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=a_opt)
        es = np.max(np.abs(t_sup - tex))

        err_gal.append(eg)
        err_up.append(eu)
        err_sup.append(es)
        print(f"{nel:<6}{eg:<14.6e}{eu:<14.6e}{es:<14.6e}")

    # 绘制双对数收敛曲线
    plt.figure(figsize=(10, 5))
    plt.loglog(nel_list, err_gal, "b--o", label="标准Galerkin")
    plt.loglog(nel_list, err_up, "r-.s", label="迎风格式")
    plt.loglog(nel_list, err_sup, "g-^", label="SUPG稳定化")
    plt.xlabel("单元数量 nel（对数坐标）")
    plt.ylabel("全局最大节点误差（对数坐标）")
    plt.title("Pe=3.0 网格加密误差收敛曲线")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show(block=True)

# ==================== 程序入口（按需注释/取消注释执行） ====================
if __name__ == "__main__":
    # 1. 运行基础作业（Pe=0.1、3，20单元，绘图+矩阵分析）
    base_case_solver()
    # 2. 运行附加网格加密收敛分析
    grid_convergence_study()