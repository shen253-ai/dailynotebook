import numpy as np
import matplotlib.pyplot as plt

# =========================
# parameters
# =========================
omega1 = 1.0
omega2 = 1.2
Delta = 1.0
S = 1.0   # 可设为1（缩放掉N）

# =========================
# energy function
# =========================
def energy(vars, g, eta):
    alpha1, alpha2, theta, phi = vars

    g1 = g
    g2 = eta * g

    E = (
        omega1 * alpha1**2
        + omega2 * alpha2**2
        + Delta * np.cos(theta)
        + 2 * g1 * alpha1 * np.sin(theta) * np.cos(phi)
        + 2 * g2 * alpha2 * np.sin(theta) * np.sin(phi)
    )
    return E

# =========================
# simple random minimizer
# =========================
def find_ground_state(g, eta):

    best_E = 1e9
    best_vars = None

    for _ in range(200):  # 多跑几次防止局部极小
        init = [
            np.random.uniform(-2,2),   # alpha1
            np.random.uniform(-2,2),   # alpha2
            np.random.uniform(0, np.pi),  # theta
            np.random.uniform(0, 2*np.pi) # phi
        ]

        vars = np.array(init)

        # 简单梯度下降（粗糙但够用）
        lr = 0.05
        for _ in range(200):
            grad = np.zeros(4)
            eps = 1e-4
            for i in range(4):
                d = np.zeros(4)
                d[i] = eps
                grad[i] = (energy(vars+d, g, eta) - energy(vars-d, g, eta)) / (2*eps)

            vars -= lr * grad

        E = energy(vars, g, eta)

        if E < best_E:
            best_E = E
            best_vars = vars

    return best_vars

# =========================
# scan phase diagram
# =========================
g_vals = np.linspace(0, 2.0, 80)
eta_vals = np.linspace(0, 2.0, 80)

alpha1_map = np.zeros((len(eta_vals), len(g_vals)))
alpha2_map = np.zeros((len(eta_vals), len(g_vals)))

for i, eta in enumerate(eta_vals):
    print("eta =", eta)
    for j, g in enumerate(g_vals):
        sol = find_ground_state(g, eta)
        alpha1_map[i,j] = sol[0]
        alpha2_map[i,j] = sol[1]

# =========================
# plot
# =========================
plt.figure(figsize=(6,5))
plt.imshow(alpha1_map,
           extent=[g_vals[0], g_vals[-1], eta_vals[0], eta_vals[-1]],
           origin='lower',
           aspect='auto')
plt.colorbar(label="alpha1")
plt.xlabel("g")
plt.ylabel("eta")
plt.title("Mode 1 Order")
plt.show()

plt.figure(figsize=(6,5))
plt.imshow(alpha2_map,
           extent=[g_vals[0], g_vals[-1], eta_vals[0], eta_vals[-1]],
           origin='lower',
           aspect='auto')
plt.colorbar(label="alpha2")
plt.xlabel("g")
plt.ylabel("eta")
plt.title("Mode 2 Order")
plt.show()