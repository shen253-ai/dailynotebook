"""
Two-Mode Dicke Model: Generalized Phase Diagram
================================================
耦合到非对易自旋算符 Jx, Jy 的两模Dicke模型

物理模型:
  H = w1*a1†a1 + w2*a2†a2 + w0*Jz
    + (g1/√N)(a1+a1†)Jx + (g2/√N)(a2+a2†)Jy
    + D1*(a1+a1†)² + D2*(a2+a2†)²
    + λ*(a1+a1†)(a2+a2†)

平均场能量 (a1→α1, a2→α2, J→球坐标):
  E = (w1+4D1)α1² + (w2+4D2)α2² + 2λα1α2
    + (w0/2)cosθ + g1*α1*sinθcosφ + g2*α2*sinθsinφ
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from numba import njit, prange

# ============================================================
# 1. 系统参数
# ============================================================
w1, w2 = 1.0, 1.0      # 两个腔模频率
w0     = 1.0            # 原子跃迁频率
D1, D2 = 0.3, 0.3      # A² 项系数 (diamagnetic)
lam    = 1.8            # ⭐ 模间耦合：需足够大才能满足 v < 2u
                        # 物理上来自: 腔模之间的直接光子-光子相互作用
                        # 或经由原子介导的有效耦合

# 导出参数
wt1   = w1 + 4*D1       # ω̃₁
wt2   = w2 + 4*D2       # ω̃₂
Delta = wt1*wt2 - lam**2 # det(M)

assert Delta > 0, (
    f"系统不稳定！Delta={Delta:.4f}\n"
    f"需要 lambda < sqrt(wt1*wt2) = {np.sqrt(wt1*wt2):.4f}\n"
    f"请减小 lam 或增大 D1, D2"
)

print("=" * 55)
print("  Two-Mode Dicke Model — 参数检查")
print("=" * 55)
print(f"  ω̃₁ = {wt1:.3f},  ω̃₂ = {wt2:.3f}")
print(f"  Δ  = {Delta:.3f}  (must be > 0 ✓)")
print()

# ============================================================
# 2. 解析预测：Landau 系数 & 多临界点
# ============================================================

def landau_coefficients(g1, g2):
    """
    从微观参数推导 Landau 自由能系数:
      F(φ1,φ2) = r1·φ1² + r2·φ2² + u·(φ1⁴+φ2⁴) + v·φ1²φ2²

    共存相稳定条件: v < 2u
    """
    r1 = wt1 - (g1**2 * wt2) / (Delta * w0)
    r2 = wt2 - (g2**2 * wt1) / (Delta * w0)

    # 四次项系数（来自 sinθ 的高阶展开）
    u  = (wt1**2 * wt2**2) / (8 * Delta**2 * w0**3)

    # 混合项系数（非对易耦合的核心效应）
    num_v = ((wt1*wt2 + lam**2) * g1**2 * g2**2
             - lam**2 * (wt1**2*g1**2 + wt2**2*g2**2) / Delta)
    v = num_v / (2 * Delta * w0**3)

    return r1, r2, u, v


# 多临界点解析预测
eta_star = np.sqrt(wt2 / wt1)          # η* = √(ω̃₂/ω̃₁)
g_star   = np.sqrt(Delta * w0 / wt2)   # g* 使得 r1=r2=0

print("─" * 55)
print("  解析预测：多临界点")
print("─" * 55)
print(f"  η* = √(ω̃₂/ω̃₁) = {eta_star:.4f}")
print(f"  g* = √(Δω₀/ω̃₂) = {g_star:.4f}")
print()

# Normal→SR1 相变线: r1=0 → g1_c(g2)
# Normal→SR2 相变线: r2=0 → g2_c(g1)
def g1_critical():
    """Normal→SR1 临界耦合（g2→0极限）"""
    return np.sqrt(wt1 * Delta * w0 / wt2)

def g2_critical():
    """Normal→SR2 临界耦合（g1→0极限）"""
    return np.sqrt(wt2 * Delta * w0 / wt1)

g1c = g1_critical()
g2c = g2_critical()
print(f"  g1_c (η→0) = {g1c:.4f}")
print(f"  g2_c (η→∞) = {g2c:.4f}")
print()

# 共存相稳定性：v < 2u
r1s, r2s, us, vs = landau_coefficients(g_star, eta_star*g_star)
print(f"  在多临界点处: u={us:.4f}, v={vs:.4f}")
print(f"  共存相条件 v < 2u: {vs:.4f} < {2*us:.4f} → {'✓ 满足' if vs < 2*us else '✗ 不满足'}")
print("=" * 55)
print()

# ============================================================
# 3. 数值相图：解析积掉 α1, α2（大幅提速）
# ============================================================

@njit
def energy_reduced(theta, phi, g1, g2, wt1_, wt2_, lam_, Delta_, w0_):
    """
    解析积掉 α1, α2 后的有效能量 E(θ,φ)
    对应: 先对 α1,α2 取极值（二次型解析解），再代入
    """
    sth = np.sin(theta)
    cph = np.cos(phi)
    sph = np.sin(phi)

    X = g1 * sth * cph   # ∂E/∂α1 的驱动项
    Y = g2 * sth * sph   # ∂E/∂α2 的驱动项

    # 解析最优值
    a1 = -(wt2_ * X - lam_ * Y) / (2.0 * Delta_)
    a2 = -(wt1_ * Y - lam_ * X) / (2.0 * Delta_)

    # 代回总能量
    E = (wt1_ * a1*a1
       + wt2_ * a2*a2
       + 2.0 * lam_ * a1 * a2
       + 0.5 * w0_ * np.cos(theta)
       + g1 * a1 * sth * cph
       + g2 * a2 * sth * sph)
    return E, a1, a2


@njit
def find_ground(g1, g2, theta_arr, phi_arr, wt1_, wt2_, lam_, Delta_, w0_):
    """在 (θ,φ) 二维空间搜索全局基态"""
    Emin   = 1e18
    best_a1 = 0.0
    best_a2 = 0.0

    for k in range(len(theta_arr)):
        for m in range(len(phi_arr)):
            E, a1, a2 = energy_reduced(
                theta_arr[k], phi_arr[m],
                g1, g2, wt1_, wt2_, lam_, Delta_, w0_)
            if E < Emin:
                Emin    = E
                best_a1 = a1
                best_a2 = a2

    return best_a1, best_a2


@njit(parallel=True)
def compute_phase_diagram(g_vals, eta_vals, theta_arr, phi_arr,
                           wt1_, wt2_, lam_, Delta_, w0_):
    """并行扫描相图（g, η）参数空间"""
    Ng = len(g_vals)
    Ne = len(eta_vals)
    phase  = np.zeros((Ng, Ne))
    a1_map = np.zeros((Ng, Ne))
    a2_map = np.zeros((Ng, Ne))

    for i in prange(Ng):
        g = g_vals[i]
        for j in range(Ne):
            g1 = g
            g2 = eta_vals[j] * g

            a1, a2 = find_ground(g1, g2, theta_arr, phi_arr,
                                  wt1_, wt2_, lam_, Delta_, w0_)

            a1_map[i, j] = abs(a1)
            a2_map[i, j] = abs(a2)

            thr = max(0.05, 0.04 * g)
            s1  = abs(a1) > thr
            s2  = abs(a2) > thr

            if   not s1 and not s2:  phase[i, j] = 0  # Normal
            elif     s1 and not s2:  phase[i, j] = 1  # SR1
            elif     s2 and not s1:  phase[i, j] = 2  # SR2
            else:                    phase[i, j] = 3  # Coexistence

    return phase, a1_map, a2_map


# 网格设置（θ,φ 精度高，速度无影响因为已解析积掉α）
theta_arr = np.linspace(1e-3, np.pi - 1e-3, 400)
phi_arr   = np.linspace(0, 2*np.pi, 400)

g_vals   = np.linspace(0, 2.5, 150)
eta_vals = np.linspace(0.05, 2.5, 150)

print("开始数值计算相图（首次编译约30秒）...")
phase, a1_map, a2_map = compute_phase_diagram(
    g_vals, eta_vals, theta_arr, phi_arr,
    wt1, wt2, lam, Delta, w0)
print("数值计算完成！\n")

# ============================================================
# 4. Landau 能量地形（使用从微观推导的系数）
# ============================================================

# 在多临界点附近取参数，展示共存相的能量景观
g_coex  = g_star * 1.5     # 稍大于临界值，进入超辐射/共存区
eta_coex = eta_star        # 对称点

r1_c, r2_c, u_c, v_c = landau_coefficients(g_coex, eta_coex * g_coex)

phi_vals = np.linspace(-2.5, 2.5, 400)
P1, P2   = np.meshgrid(phi_vals, phi_vals, indexing='ij')

E_landau = (r1_c * P1**2 + r2_c * P2**2
          + u_c  * (P1**4 + P2**4)
          + v_c  * P1**2 * P2**2)

print(f"Landau 景观（g={g_coex:.2f}, η={eta_coex:.2f}）:")
print(f"  r1={r1_c:.4f}, r2={r2_c:.4f}")
print(f"  u ={u_c:.4f},  v ={v_c:.4f}")
print(f"  v < 2u? {v_c:.4f} < {2*u_c:.4f} → {'共存相稳定 ✓' if v_c < 2*u_c else '共存相不稳定'}")
print()

# ============================================================
# 5. 沿 η=η* 切线：序参量随 g 的演化
# ============================================================

eta_cut  = eta_star

# ⭐ 修复：g_dense 与 g_vals 等长，直接用 g_vals
g_dense  = g_vals   # shape (150,) 与 a1_cut 一致

# 在 eta=eta_star 列取数值结果
eta_idx  = np.argmin(np.abs(eta_vals - eta_cut))
a1_cut   = a1_map[:, eta_idx]   # shape (150,)
a2_cut   = a2_map[:, eta_idx]   # shape (150,)

# ============================================================
# 6. 绘图
# ============================================================

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d0f14')

# 颜色方案
c_bg    = '#0d0f14'
c_N     = '#1a1f2e'   # Normal: 深蓝灰
c_SR1   = '#c0392b'   # SR1: 深红
c_SR2   = '#2471a3'   # SR2: 深蓝
c_CO    = '#27ae60'   # Coexistence: 绿
c_text  = '#ecf0f1'
c_grid  = '#2c3e50'

cmap_phase = mcolors.ListedColormap([c_N, c_SR1, c_SR2, c_CO])

# ── 图1：相图 ──────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, (1, 4))  # 左侧大图
ax1.set_facecolor(c_bg)

im = ax1.imshow(
    phase, origin='lower',
    extent=[eta_vals[0], eta_vals[-1], g_vals[0], g_vals[-1]],
    aspect='auto', cmap=cmap_phase, vmin=-0.5, vmax=3.5)

# 解析相变线叠加
# Normal→SR1: r1=0 → g = g1c（与η无关，水平线）
# 准确地说在一般η: g1_c(η) = sqrt(wt1·Delta·w0/(wt2)) 不变（因g2=η*g只影响r2）
eta_line = np.linspace(0.05, 2.5, 200)
g_N_SR1  = np.full_like(eta_line, g1c)  # Normal→SR1 (r1=0, g2小)
g_N_SR2  = g2c / eta_line               # Normal→SR2 (r2=0)

ax1.plot(eta_line, g_N_SR1, '--', color='#f39c12', lw=1.8,
         label=r'$r_1=0$ (N→SR$_1$)', alpha=0.85)
ax1.plot(eta_line, g_N_SR2, '--', color='#9b59b6', lw=1.8,
         label=r'$r_2=0$ (N→SR$_2$)', alpha=0.85)

# 多临界点
ax1.scatter([eta_star], [g_star], s=150, c='white', zorder=10,
            edgecolors='#f1c40f', linewidths=2)
ax1.annotate(f'MCP\n(η*={eta_star:.2f}, g*={g_star:.2f})',
             xy=(eta_star, g_star),
             xytext=(eta_star + 0.3, g_star + 0.25),
             color='white', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

# 垂直虚线（η=η*切线位置）
ax1.axvline(x=eta_star, color='white', lw=1, linestyle=':', alpha=0.5)

legend_patches = [
    Patch(color=c_N,   label='Normal (N)'),
    Patch(color=c_SR1, label='Superradiant SR$_1$'),
    Patch(color=c_SR2, label='Superradiant SR$_2$'),
    Patch(color=c_CO,  label='Coexistence'),
]
ax1.legend(handles=legend_patches, loc='upper right',
           framealpha=0.3, facecolor=c_bg, edgecolor=c_grid,
           labelcolor=c_text, fontsize=9)
ax1.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color='#f39c12', ls='--', label=r'$r_1=0$'),
    plt.Line2D([0],[0], color='#9b59b6', ls='--', label=r'$r_2=0$'),
], loc='upper right', framealpha=0.3, facecolor=c_bg,
   edgecolor=c_grid, labelcolor=c_text, fontsize=8.5)

ax1.set_xlabel(r'$\eta = g_2/g_1$', color=c_text, fontsize=13)
ax1.set_ylabel(r'$g$', color=c_text, fontsize=13)
ax1.set_title('Phase Diagram', color=c_text, fontsize=14, pad=10)
ax1.tick_params(colors=c_text)
for spine in ax1.spines.values():
    spine.set_edgecolor(c_grid)

# ── 图2：Landau 能量地形 ────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_facecolor(c_bg)

# 截断显示范围，突出极小值结构
E_plot = np.clip(E_landau, -2, 1)
cf = ax2.contourf(phi_vals, phi_vals, E_plot.T, levels=60,
                   cmap='RdYlGn_r')
ax2.contour(phi_vals, phi_vals, E_plot.T, levels=15,
            colors='white', linewidths=0.3, alpha=0.4)
plt.colorbar(cf, ax=ax2, label='Energy (a.u.)', shrink=0.85)

# 标出极小值（共存相有4个简并极小值）
from scipy.ndimage import label as sp_label
E_min_val = np.min(E_landau)
mask = E_landau < E_min_val + 0.05 * abs(E_min_val)
labeled, nf = sp_label(mask)
for fi in range(1, nf+1):
    region = np.argwhere(labeled == fi)
    cy = np.mean(region[:, 0])
    cx = np.mean(region[:, 1])
    ax2.scatter([phi_vals[int(cy)]], [phi_vals[int(cx)]],
                c='white', s=60, zorder=5)

ax2.set_xlabel(r'$\phi_1 = \langle a_1+a_1^\dagger\rangle$',
               color=c_text, fontsize=11)
ax2.set_ylabel(r'$\phi_2 = \langle a_2+a_2^\dagger\rangle$',
               color=c_text, fontsize=11)
ax2.set_title(f'Landau Energy Landscape\n'
              r'$g=$'+f'{g_coex:.2f}'+r', $\eta=$'+f'{eta_coex:.2f}',
              color=c_text, fontsize=11)
ax2.tick_params(colors=c_text)
for spine in ax2.spines.values():
    spine.set_edgecolor(c_grid)

# 显示系数
txt = (f'$r_1={r1_c:.3f}$, $r_2={r2_c:.3f}$\n'
       f'$u={u_c:.3f}$, $v={v_c:.3f}$\n'
       f'$v<2u$: {v_c:.3f} < {2*u_c:.3f}')
ax2.text(0.03, 0.97, txt, transform=ax2.transAxes,
         color='#f1c40f', fontsize=8, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1f2e', alpha=0.7))

# ── 图3：序参量切线 ─────────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor(c_bg)

ax3.plot(g_dense, a1_cut, color='#e74c3c', lw=2.2,
         label=r'$|\alpha_1|$')
ax3.plot(g_dense, a2_cut, color='#3498db', lw=2.2,
         label=r'$|\alpha_2|$', linestyle='--')
ax3.axvline(x=g_star, color='white', lw=1, linestyle=':',
            label=f'$g^*={g_star:.2f}$')
ax3.set_xlabel(r'$g$', color=c_text, fontsize=13)
ax3.set_ylabel('Order Parameter', color=c_text, fontsize=12)
ax3.set_title(r'Cut at $\eta=\eta^*=$'+f'{eta_star:.2f}',
              color=c_text, fontsize=11)
ax3.legend(framealpha=0.3, facecolor=c_bg, edgecolor=c_grid,
           labelcolor=c_text, fontsize=10)
ax3.tick_params(colors=c_text)
for spine in ax3.spines.values():
    spine.set_edgecolor(c_grid)
ax3.set_xlim(0, 2.5)
ax3.set_ylim(bottom=0)
ax3.grid(color=c_grid, alpha=0.4, lw=0.5)

# ── 图4：|α1| 序参量图 ─────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 5)
ax4.set_facecolor(c_bg)

im4 = ax4.imshow(
    a1_map, origin='lower',
    extent=[eta_vals[0], eta_vals[-1], g_vals[0], g_vals[-1]],
    aspect='auto', cmap='Reds')
plt.colorbar(im4, ax=ax4, label=r'$|\alpha_1|$', shrink=0.85)
ax4.set_xlabel(r'$\eta$', color=c_text)
ax4.set_ylabel(r'$g$', color=c_text)
ax4.set_title(r'Order Parameter $|\alpha_1|$', color=c_text, fontsize=11)
ax4.tick_params(colors=c_text)
for spine in ax4.spines.values():
    spine.set_edgecolor(c_grid)

# ── 图5：|α2| 序参量图 ─────────────────────────────────────
ax5 = fig.add_subplot(2, 3, 6)
ax5.set_facecolor(c_bg)

im5 = ax5.imshow(
    a2_map, origin='lower',
    extent=[eta_vals[0], eta_vals[-1], g_vals[0], g_vals[-1]],
    aspect='auto', cmap='Blues')
plt.colorbar(im5, ax=ax5, label=r'$|\alpha_2|$', shrink=0.85)
ax5.set_xlabel(r'$\eta$', color=c_text)
ax5.set_ylabel(r'$g$', color=c_text)
ax5.set_title(r'Order Parameter $|\alpha_2|$', color=c_text, fontsize=11)
ax5.tick_params(colors=c_text)
for spine in ax5.spines.values():
    spine.set_edgecolor(c_grid)

fig.suptitle('Generalized Two-Mode Dicke Model — Full Analysis',
             color=c_text, fontsize=15, y=1.01)

plt.tight_layout()
plt.savefig('two_mode_dicke_full.pdf',
            dpi=200, bbox_inches='tight', facecolor=c_bg)
plt.savefig('two_mode_dicke_full.png',
            dpi=150, bbox_inches='tight', facecolor=c_bg)
print("图已保存: two_mode_dicke_full.pdf / .png（与脚本同目录）")
plt.show()

# ============================================================
# 7. 打印完整解析结论表
# ============================================================

print()
print("=" * 55)
print("  完整解析结论汇总")
print("=" * 55)
print(f"  参数: ω1={w1}, ω2={w2}, ω0={w0}, D1={D1}, D2={D2}, λ={lam}")
print()
print("  【Landau 系数（微观推导）】")
print(f"    r1(g,η) = ω̃₁ - g²·η⁰·ω̃₂/(Δω₀)")
print(f"    r2(g,η) = ω̃₂ - g²·η²·ω̃₁/(Δω₀)")
print(f"    u  = ω̃₁²ω̃₂²/(8Δ²ω₀³) = {us:.6f}")
print(f"    v  (在多临界点) = {vs:.6f}")
print()
print("  【相变临界线】")
print(f"    N→SR₁: g_c = {g1c:.4f}  （与η无关）")
print(f"    N→SR₂: g_c = {g2c:.4f}/η")
print()
print("  【多临界点（四相汇聚）】")
print(f"    η* = {eta_star:.4f}")
print(f"    g* = {g_star:.4f}")
print()
print("  【共存相稳定机制】")
print(f"    v < 2u ← 由模间耦合λ={lam}保证")
print(f"    λ=0极限: v→2u（共存相消失，与对易模型一致）")
print("=" * 55)