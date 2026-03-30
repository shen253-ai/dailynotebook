import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
omega1 = 1.0
omega2 = 1.5
Delta = 1.0

alpha = 1.0   # controls A^2 strength

# =========================
# BdG solver
# =========================
def get_min_gap_two_mode(g, eta):
    g1 = g
    g2 = eta * g

    lam1 = alpha * g1**2 / omega1
    lam2 = alpha * g2**2 / omega2

    # BdG matrix (6x6)
    # order: (a1, a2, b, a1†, a2†, b†)
    M = np.zeros((6,6))

    # diagonal
    M[0,0] = omega1 + 2*lam1
    M[1,1] = omega2 + 2*lam2
    M[2,2] = Delta

    M[3,3] = omega1 + 2*lam1
    M[4,4] = omega2 + 2*lam2
    M[5,5] = Delta

    # coupling
    # mode 1
    gk = g1
    M[0,2] = gk
    M[2,0] = gk
    M[0,5] = gk
    M[5,0] = gk
    M[3,2] = gk
    M[2,3] = gk
    M[3,5] = gk
    M[5,3] = gk

    # mode 2
    gk = g2
    M[1,2] = gk
    M[2,1] = gk
    M[1,5] = gk
    M[5,1] = gk
    M[4,2] = gk
    M[2,4] = gk
    M[4,5] = gk
    M[5,4] = gk

    # A^2 pairing
    M[0,3] = 2*lam1
    M[3,0] = 2*lam1
    M[1,4] = 2*lam2
    M[4,1] = 2*lam2

    # BdG structure
    Sigma_z = np.diag([1,1,1,-1,-1,-1])
    H_BdG = Sigma_z @ M

    eigvals = np.linalg.eigvals(H_BdG)
    eigvals = np.real(eigvals)

    pos = eigvals[eigvals > 1e-6]
    if len(pos) == 0:
        return 0

    return np.min(pos)

# =========================
# Phase diagram
# =========================
g_vals = np.linspace(0, 2.0, 120)
eta_vals = np.linspace(0, 2.0, 120)

gap = np.zeros((len(eta_vals), len(g_vals)))

for i, eta in enumerate(eta_vals):
    for j, g in enumerate(g_vals):
        gap[i, j] = get_min_gap_two_mode(g, eta)

plt.figure(figsize=(6,5))
plt.imshow(gap,
           extent=[g_vals[0], g_vals[-1], eta_vals[0], eta_vals[-1]],
           origin='lower',
           aspect='auto')
plt.colorbar(label="Gap Ω_-")
plt.xlabel("g")
plt.ylabel("η (g2/g1)")
plt.title("Two-mode Phase Diagram")
plt.show()

# =========================
# Gap vs g (cuts)
# =========================
plt.figure()

eta_list = [0.2, 0.5, 1.0, 1.5]

for eta in eta_list:
    gaps = [get_min_gap_two_mode(g, eta) for g in g_vals]
    plt.plot(g_vals, gaps, label=f"η={eta}")

plt.xlabel("g")
plt.ylabel("gap Ω_-")
plt.legend()
plt.title("Gap vs g (cuts)")
plt.show()