"""Geometry of spectral evolution.
"""

import numpy as np
                              
N_S = 23                                     
N_Q = 380                                                                 


#-----------------------------------------------------------

def normalized_laplacian(A):
    """Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}.
    """
    A = np.asarray(A, dtype = float)
    deg = A.sum(axis = 1)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return np.eye(A.shape[0]) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])


def combinatorial_laplacian(A):
    """Combinatorial Laplacian: L = D - A.
    """
    A = np.asarray(A, dtype = float)
    return np.diag(A.sum(axis = 1)) - A


def esd(A, kind = "normalized"):
    """Empirical Spectral Density (ESD).
    """
    if kind == "normalized":
        L = normalized_laplacian(A) 
    else:
        L = combinatorial_laplacian(A)
    
    lam = np.linalg.eigvalsh(L) 
    return np.clip(lam, 0.0, None) if kind == "normalized" else lam



def esd_trajectory(adjacencies, kind = "normalized"):
    """Stack the ESDs of a temporal graph into a (T, n) array.
    """
    return np.array([esd(A, kind = kind) for A in adjacencies])



def quantile_grid(n_q = N_Q):
    """Midpoint quantile grid q_j = (j + 1/2)/N_Q.
    """
    return (np.arange(n_q) + 0.5) / n_q


def quantile_function(atoms, q):
    """Quantile function F^{-1}(q) = inf{x : F(x) >= q}.
    """
    atoms = np.sort(np.asarray(atoms, dtype=float))
    n = atoms.size
    idx = np.clip(np.floor(np.asarray(q) * n).astype(int), 0, n - 1)
    return atoms[idx]



def w2_squared(atoms_a, atoms_b, n_q = N_Q):
    """W_2^2 = int_0^1 |F_mu^{-1} - F_nu^{-1}|^2 dq.
    """
    a = np.sort(np.asarray(atoms_a, dtype = float))
    b = np.sort(np.asarray(atoms_b, dtype = float))
    if a.size == b.size:
        return float(np.mean((a - b) ** 2))
    q = quantile_grid(n_q)
    return float(np.mean((quantile_function(a, q) - quantile_function(b, q)) ** 2)) 


def w2(atoms_a, atoms_b, n_q = N_Q):
    return float(np.sqrt(max(w2_squared(atoms_a, atoms_b, n_q = n_q), 0.0)))


def geodesic(atoms_a, atoms_b, s):
    a = np.sort(np.asarray(atoms_a, dtype = float))
    b = np.sort(np.asarray(atoms_b, dtype = float))
    s = np.asarray(s, dtype=float)
    return (1.0 - s[..., None]) * a + s[..., None] * b



def spectral_speed(traj, dt = 1.0):
    """Discrete spectral speed v(t) = W_2(mu_t, mu_{t+1}) / dt
    """
    return np.array([w2(traj[t], traj[t + 1]) / dt for t in range(len(traj) - 1)])


def spectral_length(traj, dt = 1.0):
    """Total spectral variation, the length of the discrete curve.
    """
    return float(spectral_speed(traj, dt=dt).sum() * dt)


