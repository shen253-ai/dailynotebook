import numpy as np
import matplotlib.pyplot as plt

# parameters
omega = 1.0
Delta = 1.0

def get_min_gap(g, lam):
    # BdG matrix M
    M = np.array([
        [omega + 2*lam,      g,            2*lam,         g],
        [g,                  Delta,        g,             0],
        [2*lam,              g,            omega + 2*lam, g],
        [g,                  0,            g,             Delta]
    ], dtype=float)

    # Sigma_z
    Sigma_z = np.diag([1, 1, -1, -1])

    # BdG Hamiltonian
    H_BdG = Sigma_z @ M

    # eigenvalues
    eigvals = np.linalg.eigvals(H_BdG)

    # take real part (numerical stability)
    eigvals = np.real(eigvals)

    # keep positive frequencies
    positive = eigvals[eigvals > 1e-6]

    if len(positive) == 0:
        return 0

    return np.min(positive)


# grid
g_vals = np.linspace(0, 2.0, 200)
lam_vals = np.linspace(0, 1.5, 200)

gap = np.zeros((len(lam_vals), len(g_vals)))

# compute phase diagram
for i, lam in enumerate(lam_vals):
    for j, g in enumerate(g_vals):
        gap[i, j] = get_min_gap(g, lam)

# plot
plt.figure(figsize=(6,5))
plt.imshow(gap, extent=[g_vals[0], g_vals[-1], lam_vals[0], lam_vals[-1]],
           origin='lower', aspect='auto')
plt.colorbar(label="Gap Ω_-")
plt.plot(g_vals, g_vals**2/omega, 'r--', label='no-go')
plt.legend()
plt.xlabel("g")
plt.ylabel("λ")
plt.title("Phase Diagram (Gap)")
plt.show()
