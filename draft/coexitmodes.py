import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ===== 参数 =====
w1, w2 = 1.0, 1.0
w0 = 1.0
D1, D2 = 0.3, 0.3
lam = 0.5

# ===== 网格 =====
a_range = np.linspace(-2, 2, 80)
theta_range = np.linspace(0, np.pi, 60)
phi_range = np.linspace(0, 2*np.pi, 60)

# ===== 能量函数（JIT）=====
@njit
def energy(a1, a2, theta, phi, g1, g2):
    E = (w1 + 4*D1)*a1*a1 + (w2 + 4*D2)*a2*a2
    E += 2*lam*a1*a2
    E += 0.5*w0*np.cos(theta)
    E += g1*a1*np.sin(theta)*np.cos(phi)
    E += g2*a2*np.sin(theta)*np.sin(phi)
    return E

# ===== 单点全局搜索（不并行！关键）=====
@njit
def find_ground_fast(g1, g2, a_range, theta_range, phi_range):

    Emin = 1e18
    best_a1, best_a2 = 0.0, 0.0

    for i in range(len(a_range)):
        a1 = a_range[i]

        for j in range(len(a_range)):
            a2 = a_range[j]

            for k in range(len(theta_range)):
                th = theta_range[k]

                for m in range(len(phi_range)):
                    ph = phi_range[m]

                    E = energy(a1, a2, th, ph, g1, g2)

                    if E < Emin:
                        Emin = E
                        best_a1 = a1
                        best_a2 = a2

    return best_a1, best_a2

# ===== 参数扫描（这里并行！）=====
@njit(parallel=True)
def compute_phase(g_vals, eta_vals, a_range, theta_range, phi_range):

    phase = np.zeros((len(g_vals), len(eta_vals)))

    for i in prange(len(g_vals)):  # ⭐ 多核在这里
        g = g_vals[i]

        for j in range(len(eta_vals)):
            eta = eta_vals[j]

            g1 = g
            g2 = eta * g

            a1, a2 = find_ground_fast(g1, g2, a_range, theta_range, phi_range)

            # ===== 相分类 =====
            if abs(a1) < 0.1 and abs(a2) < 0.1:
                phase[i, j] = 0
            elif abs(a1) > 0.1 and abs(a2) < 0.1:
                phase[i, j] = 1
            elif abs(a2) > 0.1 and abs(a1) < 0.1:
                phase[i, j] = 2
            else:
                phase[i, j] = 3

    return phase

# ===== 主程序 =====
g_vals = np.linspace(0, 2.5, 60)
eta_vals = np.linspace(0.1, 2.0, 60)

print("开始计算（第一次会编译，稍等30秒左右）...")
phase = compute_phase(g_vals, eta_vals, a_range, theta_range, phi_range)
print("计算完成！")

# ===== 画图 =====
plt.figure(figsize=(6,5))
plt.imshow(phase, origin='lower',
           extent=[eta_vals[0], eta_vals[-1], g_vals[0], g_vals[-1]],
           aspect='auto')

plt.colorbar(label='Phase (0=N,1=M1,2=M2,3=Coexist)')
plt.xlabel("eta (g2/g1)")
plt.ylabel("g")
plt.title("Optimized Phase Diagram")

plt.show()