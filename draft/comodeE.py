import numpy as np
import matplotlib.pyplot as plt

# ===== 参数（你可以调）=====
r1 = -1.0
r2 = -1.0
u = 1.0
v = 3.0   # ⭐ 关键：大于 2u

# ===== 网格 =====
a_vals = np.linspace(-2, 2, 300)

# ⭐ 向量化网格
A1, A2 = np.meshgrid(a_vals, a_vals, indexing='ij')

# ===== 能量函数（向量化版本）=====
E = (
    r1 * A1**2
    + r2 * A2**2
    + u * (A1**4 + A2**4)
    + v * (A1**2) * (A2**2)
)

# ===== 找全局最小值 =====
idx = np.unravel_index(np.argmin(E), E.shape)
a1_min = a_vals[idx[0]]
a2_min = a_vals[idx[1]]

print("Ground state:")
print("alpha1 =", a1_min)
print("alpha2 =", a2_min)

# ===== 画能量地形 =====
plt.figure(figsize=(6,5))
plt.contourf(a_vals, a_vals, E.T, levels=60)
plt.colorbar(label="Energy")

plt.scatter(a1_min, a2_min, c='red', s=80, label='Ground state')

plt.xlabel("alpha1")
plt.ylabel("alpha2")
plt.title("Landau Energy Landscape")

plt.legend()
plt.show()