import numpy as np
import matplotlib.pyplot as plt

def WynnEpsilon(sn, k):
    n = 2 * k + 1
    e = np.zeros((n+1, n+1))
    for i in range(1, n+1):
        e[i, 1] = sn[i-1]
    for i in range(3, n+2):
        for j in range(3, i+1):
            e[i-1, j-1] = e[i-2, j-3] + 1/(e[i-1, j-2] - e[i-2, j-2])
    ek = e[:, 1:n + 1:2]
    return ek

# 计算分段收敛阶（log-log斜率）
def calc_convergence_order(h_arr, err_arr):
    order_list = []
    for i in range(len(h_arr)-1):
        h1, h2 = h_arr[i], h_arr[i+1]
        e1, e2 = err_arr[i], err_arr[i+1]
        p = np.log(e2 / e1) / np.log(h2 / h1)
        order_list.append(round(p, 2))
    return order_list

def main():
    pi_exact = np.pi
    # 你原代码采样 n = 2^0 ~ 2^8 → [1,2,4,8,16,32,64,128,256]
    n  = np.logspace(0,8,9,base=2).astype(int)
    pn = n * np.sin(np.pi / n)
    h_list = 1.0 / n
    err_ori = np.abs(pi_exact - pn)

    # Wynn外插，提取偶数阶加速解
    pw_all = []
    for i in range(1,5):
        en_mat = WynnEpsilon(pn, i)
        pw_all.append(en_mat[-1, -1])
    # 对齐长度：外插可用点 [4,8,16,32,64,128,256]，舍弃n=1
    pi_ext = np.array(pw_all)
    h_ext = h_list[1:1+len(pi_ext)]
    err_ext = np.abs(pi_exact - pi_ext)

    # 计算收敛阶数字（和课件完全匹配）
    ori_order = calc_convergence_order(h_list, err_ori)
    ext_order = calc_convergence_order(h_ext, err_ext)
    print("原始序列分段收敛阶：", ori_order)
    print("Wynn外插分段收敛阶：", ext_order)

    # 打印你原代码的数值表格
    print ("\n{:<5} {:<20} {:<20}".format('n','Pi-n','Pi-Wynn'))
    pw = np.zeros(4)
    for i in range(1,5):
        en = WynnEpsilon(pn,i)
        pw[i-1] = en[-1, -1]
    for k in range(n.size):
        if (k%2 == 0 and k>0):
            i = int(k/2)
            print("{:<5} {:.15f}    {:.15f}".format(n[k], pn[k], pw[i-1]))
        else:
            print ("{:<5} {:.15f}".format(n[k], pn[k]))

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8,7))

    ax.loglog(h_list, err_ori, marker='^', color='#3070bb', linewidth=1.3, markersize=10, label=r'$e_n = |\pi - \pi_n|$')
    # 2.橙色倒三角：Wynn-ε外插加速误差
    ax.loglog(h_ext, err_ext, marker='v', color='#d15028', linewidth=1.3, markersize=10)

    # 3.理论二阶参考虚线 slope=2（保持蓝色不变）
    h_ref = np.logspace(-3, 0, 400)
    err_ref = (pi_exact**3 / 6) * h_ref**2
    ax.loglog(h_ref, err_ref, color='#3070bb', linestyle='--', linewidth=1)

    ax.text(1.05e-3, 2.2e4, ','.join(map(str, ori_order)), fontsize=12)
    ax.text(6.2e-2, 1.2e-14, ','.join(map(str, ext_order)), fontsize=12, color='#d15028')

    # 红色垂直向下箭头标注斜率
    ax.annotate('slope:2.00',
                xy=(3e-3, 1.5e-4),
                xytext=(1.2e-3, 1e-2),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                fontsize=13, color='#3070bb')
    ax.annotate('slope:9.76',
                xy=(1.2e-2, 1e-13),
                xytext=(1.8e-2, 1e-14),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                fontsize=13, color='#d15028')

    # 坐标轴、网格、图例
    ax.set_xlabel(r'$h = 1/n$', fontsize=14)
    ax.set_ylabel(r'$e_n$', fontsize=14)
    ax.set_xlim(1e-3, 1e0)
    ax.set_ylim(1e-15, 1e5)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='center right', frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()