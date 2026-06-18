import numpy as np

class TrussModel:
    def __init__(self, data):
        self.data = data
        self.dim = self.data["dimension"]
        self.n_node = self.data["n_node"]
        self.n_elem = self.data["n_elem"]
        self.ndof_node = self.dim
        self.ndof_total = self.n_node * self.ndof_node

        self.node_coords = np.array(self.data["node_coords"], dtype=float)
        self.IEN = np.array(self.data["IEN"], dtype=int)
        self.elem_mat = np.array(self.data["elem_material"], dtype=float)
        self.bc_list = self.data["boundary_cond"]
        self.load_list = self.data["node_load"]

        self.LM = self._build_LM()
        self.K_global = np.zeros((self.ndof_total, self.ndof_total), dtype=float)
        self.u_global = np.zeros(self.ndof_total, dtype=float)
        self.F_global = np.zeros(self.ndof_total, dtype=float)
        for dof, val in self.load_list:
            self.F_global[dof] = val

    def _get_global_dof(self, node_idx, local_dof):
        return node_idx * self.ndof_node + local_dof

    def _build_LM(self):
        ndof_elem = 2 * self.ndof_node
        LM = np.zeros((ndof_elem, self.n_elem), dtype=int)
        for e in range(self.n_elem):
            ni, nj = self.IEN[e]
            for ld in range(self.ndof_node):
                LM[ld, e] = self._get_global_dof(ni, ld)
            for ld in range(self.ndof_node):
                LM[ld + self.ndof_node, e] = self._get_global_dof(nj, ld)
        return LM

def calc_elem_stiffness(ni_coord, nj_coord, E, A, dim):
    dx = nj_coord[0] - ni_coord[0]
    if dim == 1:
        L = dx
        if abs(L) < 1e-10:
            raise ValueError("单元两节点重合，长度为0！")
        Ke = (E * A / L) * np.array([[1, -1], [-1, 1]])
    elif dim == 2:
        dy = nj_coord[1] - ni_coord[1]
        L = np.sqrt(dx**2 + dy**2)
        if abs(L) < 1e-10:
            raise ValueError("单元两节点重合，长度为0！")
        c = dx / L
        s = dy / L
        T_mat = np.array([[c, s, 0, 0], [0, 0, c, s]])
        k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
        Ke = T_mat.T @ k_local @ T_mat
    else:
        raise NotImplementedError("仅支持1D/2D桁架")
    return Ke, L

def assemble_global_K(model):
    n_elem = model.n_elem
    ndof_elem = model.LM.shape[0]
    for e in range(n_elem):
        ni, nj = model.IEN[e]
        E, A = model.elem_mat[e]
        ni_c = model.node_coords[ni]
        nj_c = model.node_coords[nj]
        Ke, _ = calc_elem_stiffness(ni_c, nj_c, E, A, model.dim)
        for iloc in range(ndof_elem):
            iglob = model.LM[iloc, e]
            for jloc in range(ndof_elem):
                jglob = model.LM[jloc, e]
                model.K_global[iglob, jglob] += Ke[iloc, jloc]
    is_symmetric = np.allclose(model.K_global, model.K_global.T, atol=1e-8)
    print(f"【组装校验】总体刚度矩阵对称：{is_symmetric}")
    return model.K_global

def solve_truss(model):
    K = model.K_global.copy()
    F = model.F_global.copy()
    ndof_total = model.ndof_total

    known_dof = [dof for dof, val in model.bc_list]
    known_val = [val for dof, val in model.bc_list]
    unknown_dof = [d for d in range(ndof_total) if d not in known_dof]

    K_red = K[np.ix_(unknown_dof, unknown_dof)]
    F_red = F[unknown_dof]
    for idx, dof in enumerate(known_dof):
        disp_val = known_val[idx]
        F_red -= K[np.ix_(unknown_dof, [dof])].flatten() * disp_val

    u_unknown = np.linalg.solve(K_red, F_red)
    u_full = np.zeros(ndof_total)
    u_full[unknown_dof] = u_unknown
    for idx, dof in enumerate(known_dof):
        u_full[dof] = known_val[idx]
    model.u_global = u_full

    R_full = K @ u_full - F
    bc_reaction = {dof: R_full[dof] for dof in known_dof}
    return u_full, bc_reaction

def post_process(model):
    res_list = []
    ndof_elem = model.LM.shape[0]
    for e in range(model.n_elem):
        ni, nj = model.IEN[e]
        E, A = model.elem_mat[e]
        ni_c = model.node_coords[ni]
        nj_c = model.node_coords[nj]
        Ke, L = calc_elem_stiffness(ni_c, nj_c, E, A, model.dim)

        u_elem = np.zeros(ndof_elem)
        for iloc in range(ndof_elem):
            iglob = model.LM[iloc, e]
            u_elem[iloc] = model.u_global[iglob]

        if model.dim == 1:
            N = (E * A / L) * (u_elem[1] - u_elem[0])
        elif model.dim == 2:
            dx = nj_c[0] - ni_c[0]
            dy = nj_c[1] - ni_c[1]
            c = dx / L
            s = dy / L
            delta_u = c*(u_elem[2]-u_elem[0]) + s*(u_elem[3]-u_elem[1])
            N = (E * A / L) * delta_u
        else:
            N = 0.0
        stress = N / A
        res_list.append({
            "elem_id": e,
            "node_i": ni,
            "node_j": nj,
            "length": L,
            "axial_force": N,
            "stress": stress
        })
    return res_list

# ===================== PPT标准二维算例：E=1，A=1，Fx=10 =====================
truss_2d_data = {
    "dimension": 2,
    "n_node": 3,
    "n_elem": 2,
    "node_coords": [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
    "IEN": [[0, 2], [1, 2]],
    "elem_material": [[1.0, 1.0], [1.0, 1.0]],  # E1=E2=1, A1=A2=1
    "boundary_cond": [[0,0.0], [1,0.0], [2,0.0], [3,0.0]],
    "node_load": [[4, 10.0]]  # 节点3水平自由度4，荷载Fx=10
}

def run_calc(model_data, title):
    print(f"\n【载荷校验】node_load = {model_data['node_load']}")
    print("=" * 60)
    print(f"开始计算模型：{title}")
    model = TrussModel(model_data)
    assemble_global_K(model)
    u_full, reaction = solve_truss(model)
    elem_results = post_process(model)

    print("\n【全局节点位移】")
    for i, val in enumerate(u_full):
        print(f"全局自由度 {i:2d} : {val:.6f}")

    print("\n【约束反力】")
    for dof, val in reaction.items():
        print(f"约束自由度 {dof:2d} 反力: {val:.6f}")

    print("\n【单元轴力&应力（匹配PPT理论值）】")
    for info in elem_results:
        print(f"单元{info['elem_id']}({info['node_i']}-{info['node_j']}) "
              f"轴力={info['axial_force']:.6f}, 应力={info['stress']:.6f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_calc(truss_2d_data, "PPT标准二维两杆桁架(E=1,A=1,Fx=10)")