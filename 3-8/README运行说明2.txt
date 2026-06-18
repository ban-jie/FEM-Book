运行方法
 
1. 安装依赖： pip install numpy matplotlib 

2. 直接运行脚本： python advection_diff_fem.py 
 
程序功能对应作业要求
 
1. 单元矩阵函数 element_matrix 
严格匹配公式：

\boldsymbol{K}_e = \frac{\bar{\kappa}}{l_e}\begin{bmatrix}1 & -1 \\ -1 & 1\end{bmatrix} + \frac{v}{2}\begin{bmatrix}-1 & 1 \\ -1 & 1\end{bmatrix},\quad \bar{\kappa}=\kappa+\alpha\frac{vl_e}{2}

 
- \alpha=0：标准Galerkin

- \alpha=1：迎风格式人工扩散

- \alpha=\coth(Pe)-1/Pe：SUPG最优稳定参数
 
2. 网格与组装
固定 nel=20 等长单元，循环叠加单元矩阵至全局，对角置1法施加Dirichlet边界\theta(0)=0,\theta(1)=1。

3. 两组工况自动计算
Pe=0.1（扩散占优）、Pe=3.0（对流占优），由Pe=\dfrac{vl_e}{2\kappa}自动反推扩散系数\kappa。

4. 绘图输出
每组Pe生成一张对比图，包含：精确解、Galerkin、迎风、SUPG四条曲线。

5. 误差计算
\text{max\_error}=\max|\theta_{\text{numerical}}-\theta_{\text{exact}}|
打印结构化误差表格。

6. 矩阵性质分析（Pe=3）
检验全局刚度矩阵对称性、特征值、正定性，输出结论解释振荡根源。
 
运行结果说明
 
1. Pe=0.1
 
- Galerkin、SUPG曲线贴近精确解，无振荡；迎风轻微抹平边界，误差略大。
 
2. Pe=3（对流占优）
 
- 标准Galerkin曲线剧烈上下振荡，误差极大；矩阵非正定、不对称；

- 迎风无振荡，但右侧边界层被抹平，数值扩散严重；

- SUPG曲线光滑，完美贴合精确解，误差最小，平衡稳定与精度。
 
作业报告配套矩阵公式（直接复制进文档）
 
1 标准Galerkin单元矩阵（α=0，\bar{\kappa}=\kappa）
 

\boldsymbol{K}_e
=
\frac{\kappa}{l_e}
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}
+
\frac{v}{2}
\begin{bmatrix}
-1 & 1 \\
-1 & 1
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\kappa}{l_e}-\dfrac{v}{2} & -\dfrac{\kappa}{l_e}+\dfrac{v}{2} \\[6pt]
-\dfrac{\kappa}{l_e}-\dfrac{v}{2} & \dfrac{\kappa}{l_e}+\dfrac{v}{2}
\end{bmatrix}

 
2 统一稳定化单元矩阵（带人工扩散\bar{\kappa}）
 
修正扩散系数：
\bar{\kappa} = \kappa + \alpha \cdot \frac{v l_e}{2}
单元矩阵：

\boldsymbol{K}_e
=
\frac{\bar{\kappa}}{l_e}
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}
+
\frac{v}{2}
\begin{bmatrix}
-1 & 1 \\
-1 & 1
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\bar{\kappa}}{l_e}-\dfrac{v}{2} & -\dfrac{\bar{\kappa}}{l_e}+\dfrac{v}{2} \\[6pt]
-\dfrac{\bar{\kappa}}{l_e}-\dfrac{v}{2} & \dfrac{\bar{\kappa}}{l_e}+\dfrac{v}{2}
\end{bmatrix}

 
3 SUPG稳定参数
 
\alpha_{\text{opt}} = \coth(Pe) - \frac{1}{Pe},\quad Pe=\frac{v l_e}{2\kappa}