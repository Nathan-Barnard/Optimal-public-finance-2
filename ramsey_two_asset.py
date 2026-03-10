from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_NEG_INF = -1.0e300
_EPS = 1.0e-12


@dataclass(frozen=True)
class Calibration:
    rho: float = 0.05
    gamma: float = 0.8
    chi: float = 0.65
    delta: float = 0.05
    g: float = 0.02
    sigma: float = 0.15
    lam: float = 0.3
    I0: float = 0.32
    I1: float = 0.52
    tau_max: float = 0.45
    transfer_floor: float = -0.20
    transfer_cap: float = 1.20
    k_min: float = 0.50
    k_max: float = 4.00
    L_min: float = -0.45
    L_max: float = 1.10
    k_points: int = 19
    L_points: int = 19
    tau_points: int = 9
    H_points: int = 9
    transfer_points: int = 11
    interior_buffer: int = 1
    omega_init: float = 0.08
    zeta_omega: float = 0.50
    eta_policy: float = 0.75
    eps_c: float = 1.0e-8
    eps_drift: float = 1.0e-12
    max_outer: int = 25
    max_inner: int = 25
    tol_outer: float = 1.0e-6
    tol_policy: float = 1.0e-6
    peel_steps: int = 6
    coarse_init: bool = True

    def with_gamma(self, gamma: float) -> "Calibration":
        return replace(self, gamma=float(gamma))


@dataclass(frozen=True)
class ExperimentSpec:
    output_dir: str = "outputs"
    run_baseline: bool = True
    run_comparative_statics: bool = True
    run_log_benchmark: bool = True
    eis_values: Tuple[float, ...] = (1.50, 1.00, 0.70)
    plot_2d_heatmaps: bool = True
    plot_slices: bool = True
    verbose: bool = False


@dataclass
class _Grid:
    k: np.ndarray
    L: np.ndarray
    interior_buffer: int = 1

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float).reshape(-1)
        self.L = np.asarray(self.L, dtype=float).reshape(-1)
        if self.k.size < 3 or self.L.size < 3:
            raise ValueError("Grid must have at least 3 points in each dimension.")
        if not np.all(np.diff(self.k) > 0.0):
            raise ValueError("k grid must be strictly increasing.")
        if not np.all(np.diff(self.L) > 0.0):
            raise ValueError("L grid must be strictly increasing.")
        self.Nx = int(self.k.size)
        self.Ny = int(self.L.size)
        self.dk = float(self.k[1] - self.k[0])
        self.dL = float(self.L[1] - self.L[0])
        if self.Nx > 2 and not np.allclose(np.diff(self.k), self.dk):
            raise ValueError("k grid must be uniform.")
        if self.Ny > 2 and not np.allclose(np.diff(self.L), self.dL):
            raise ValueError("L grid must be uniform.")
        self.interior_mask = _make_interior_mask(self.Nx, self.Ny, self.interior_buffer)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.Nx, self.Ny)


@dataclass(frozen=True)
class _Prim:
    tau_grid: np.ndarray
    H_grid: np.ndarray
    T_grid: np.ndarray
    tau_min: float
    tau_max: float
    T_min: float
    T_max: float


@dataclass(frozen=True)
class _ActiveIndex:
    idx_full: np.ndarray
    inv_full: np.ndarray
    n_active: int


@dataclass(frozen=True)
class _NodeDriftFlow:
    kdot: float
    Ldot: float
    flow: float


def _make_interior_mask(nx: int, ny: int, buffer: int) -> np.ndarray:
    if buffer < 0:
        raise ValueError("buffer must be nonnegative.")
    if buffer == 0:
        return np.ones((nx, ny), dtype=bool)
    if nx <= 2 * buffer or ny <= 2 * buffer:
        raise ValueError("Grid too small for requested interior buffer.")
    out = np.ones((nx, ny), dtype=bool)
    out[:buffer, :] = False
    out[-buffer:, :] = False
    out[:, :buffer] = False
    out[:, -buffer:] = False
    return out


def _iter_nodes_where(mask: np.ndarray) -> Iterator[Tuple[int, int]]:
    ii, jj = np.where(mask)
    for i, j in zip(ii, jj):
        yield int(i), int(j)


def _flatten(node: Tuple[int, int], grid: _Grid) -> int:
    i, j = node
    return i * grid.Ny + j


def _unflatten(idx: int, grid: _Grid) -> Tuple[int, int]:
    return int(idx // grid.Ny), int(idx % grid.Ny)


def _empty_policy_like(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    return {
        "tau": np.full(shape, np.nan, dtype=float),
        "H": np.full(shape, np.nan, dtype=float),
        "T": np.full(shape, np.nan, dtype=float),
    }


def _mask_policy(policy: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(value, dtype=float).copy() for key, value in policy.items()}
    for key in out:
        out[key][~mask] = np.nan
    return out


def _build_grid(cal: Calibration) -> _Grid:
    return _Grid(
        k=np.linspace(cal.k_min, cal.k_max, cal.k_points),
        L=np.linspace(cal.L_min, cal.L_max, cal.L_points),
        interior_buffer=cal.interior_buffer,
    )


def _build_prim(cal: Calibration) -> _Prim:
    return _Prim(
        tau_grid=np.linspace(0.0, cal.tau_max, cal.tau_points),
        H_grid=np.linspace(0.0, cal.k_max, cal.H_points),
        T_grid=np.linspace(cal.transfer_floor, cal.transfer_cap, cal.transfer_points),
        tau_min=0.0,
        tau_max=cal.tau_max,
        T_min=cal.transfer_floor,
        T_max=cal.transfer_cap,
    )


def _phi(I: float) -> float:
    I = float(np.clip(I, _EPS, 1.0 - _EPS))
    return float(np.exp(-I * np.log(I) - (1.0 - I) * np.log(1.0 - I)))


def _regime_I(cal: Calibration, s: int) -> float:
    return float(cal.I1 if int(s) == 1 else cal.I0)


def _production_block(cal: Calibration, s: int, k: float) -> Tuple[float, float, float, float]:
    if k <= 0.0 or not np.isfinite(k):
        return 0.0, 0.0, -cal.delta, cal.sigma
    I = _regime_I(cal, s)
    Y = _phi(I) * (k ** I)
    w = (1.0 - I) * Y
    r_k = I * (Y / k) - cal.delta
    sigma_K = cal.sigma
    return float(Y), float(w), float(r_k), float(sigma_K)


def _crra_utility(c: float, gamma: float) -> float:
    if not np.isfinite(c) or c <= 0.0:
        return float("-inf")
    if np.isclose(gamma, 1.0):
        return float(np.log(max(c, _EPS)))
    return float(np.exp((1.0 - gamma) * np.log(max(c, _EPS))) / (1.0 - gamma))


def _owner_wealth(k: float, L: float) -> float:
    return float(k + L)


def _private_risky_share(k: float, L: float, H: float) -> float:
    W = _owner_wealth(k, L)
    if W <= _EPS:
        return float("nan")
    return float((k - H) / W)


def _safe_rate(cal: Calibration, s: int, k: float, L: float, H: float, tau: float) -> float:
    _, _, r_k, sigma_K = _production_block(cal, s, k)
    pi_mc = _private_risky_share(k, L, H)
    if not np.isfinite(pi_mc):
        return float("nan")
    tau = float(np.clip(tau, 0.0, cal.tau_max))
    return float(r_k - cal.gamma * (1.0 - tau) * (sigma_K ** 2) * pi_mc)


def _control_bounds(cal: Calibration, s: int, k: float, L: float, w: float) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    tau_bounds = (0.0, cal.tau_max)
    T_lo = max(cal.transfer_floor, -w + cal.eps_c)
    T_bounds = (T_lo, cal.transfer_cap)
    H_bounds = (max(0.0, -L), k)
    return tau_bounds, T_bounds, H_bounds


def _static_feasible(cal: Calibration, s: int, k: float, L: float, tau: float, H: float, T: float, omega: float) -> bool:
    if not all(np.isfinite(x) for x in (k, L, tau, H, T, omega)):
        return False
    Y, w, _, _ = _production_block(cal, s, k)
    del Y
    tau_bounds, T_bounds, H_bounds = _control_bounds(cal, s, k, L, w)
    if tau < tau_bounds[0] or tau > tau_bounds[1]:
        return False
    if T < T_bounds[0] or T > T_bounds[1]:
        return False
    if H < H_bounds[0] or H > H_bounds[1]:
        return False
    W = _owner_wealth(k, L)
    if W <= 0.0 or L < -k:
        return False
    c_w = w + T
    c_k = omega * W
    return bool(c_w > cal.eps_c and c_k > cal.eps_c)


def _node_flow_and_drift(
    cal: Calibration,
    s: int,
    k: float,
    L: float,
    tau: float,
    H: float,
    T: float,
    omega: float,
    *,
    require_feasible: bool = True,
) -> Tuple[float, float, float]:
    if require_feasible and not _static_feasible(cal, s, k, L, tau, H, T, omega):
        return float("nan"), float("nan"), float("nan")
    Y, w, r_k, _ = _production_block(cal, s, k)
    r_f = _safe_rate(cal, s, k, L, H, tau)
    W = _owner_wealth(k, L)
    if not all(np.isfinite(x) for x in (Y, w, r_k, r_f, W)):
        return float("nan"), float("nan"), float("nan")
    c_w = w + T
    c_k = omega * W
    if c_w <= cal.eps_c or c_k <= cal.eps_c:
        return float("nan"), float("nan"), float("nan")
    flow = cal.chi * _crra_utility(c_w, cal.gamma) + (1.0 - cal.chi) * _crra_utility(c_k, cal.gamma)
    kdot = Y - c_w - c_k - (cal.delta + cal.g) * k
    B = L + H
    tax_base = (k - H) * r_k + r_f * B
    Ldot = r_f * B + T - H * r_k - tau * tax_base
    return float(flow), float(kdot), float(Ldot)


def _primitive_feasible_mask(grid: _Grid) -> np.ndarray:
    K, LL = np.meshgrid(grid.k, grid.L, indexing="ij")
    return (K >= 0.0) & (LL >= -K) & ((K + LL) >= 0.0)


def _inward_one_cell_node(active: np.ndarray, node: Tuple[int, int], kdot: float, Ldot: float, eps: float) -> bool:
    if not (np.isfinite(kdot) and np.isfinite(Ldot)):
        return False
    i, j = node
    nx, ny = active.shape
    if kdot > eps and i == nx - 1:
        return False
    if kdot < -eps and i == 0:
        return False
    if Ldot > eps and j == ny - 1:
        return False
    if Ldot < -eps and j == 0:
        return False
    if kdot > eps and not active[i + 1, j]:
        return False
    if kdot < -eps and not active[i - 1, j]:
        return False
    if Ldot > eps and not active[i, j + 1]:
        return False
    if Ldot < -eps and not active[i, j - 1]:
        return False
    return bool(active[i, j])


def _inward_one_cell(active: np.ndarray, kdot: np.ndarray, Ldot: np.ndarray, eps: float) -> np.ndarray:
    out = np.zeros_like(active, dtype=bool)
    for node in _iter_nodes_where(active):
        out[node] = _inward_one_cell_node(active, node, float(kdot[node]), float(Ldot[node]), eps)
    return out


def _coarse_candidates(cal: Calibration, prim: _Prim, k: float, L: float, w: float) -> Iterable[Tuple[float, float, float]]:
    tau_vals = np.unique(np.clip(np.array([0.0, 0.5 * cal.tau_max, cal.tau_max]), 0.0, cal.tau_max))
    T_lo = max(cal.transfer_floor, -w + cal.eps_c)
    T_vals = np.unique(np.clip(np.array([T_lo, 0.0, cal.transfer_cap]), T_lo, cal.transfer_cap))
    H_lo = max(0.0, -L)
    H_vals = np.unique(np.clip(np.array([H_lo, 0.5 * (H_lo + k), k]), H_lo, k))
    for tau in tau_vals:
        for H in H_vals:
            for T in T_vals:
                yield float(tau), float(H), float(T)


def _viability_peel_warm(grid: _Grid, cal: Calibration, omega: np.ndarray, s: int, base_active: np.ndarray, steps: int) -> np.ndarray:
    active = np.asarray(base_active, dtype=bool).copy()
    for _ in range(max(1, steps)):
        new_active = active.copy()
        for node in _iter_nodes_where(active):
            i, j = node
            k = float(grid.k[i])
            L = float(grid.L[j])
            _, w, _, _ = _production_block(cal, s, k)
            feasible = False
            for tau, H, T in _coarse_candidates(cal, _build_prim(cal), k, L, w):
                omg = float(omega[node])
                if not _static_feasible(cal, s, k, L, tau, H, T, omg):
                    continue
                _, kd, Ld = _node_flow_and_drift(cal, s, k, L, tau, H, T, omg)
                if _inward_one_cell_node(active, node, kd, Ld, cal.eps_drift):
                    feasible = True
                    break
            if not feasible:
                new_active[node] = False
        if np.array_equal(new_active, active):
            break
        active = new_active
    return active


def _masked_upwind_derivatives(J: np.ndarray, active: np.ndarray, grid: _Grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Jk_f = np.full_like(J, np.nan, dtype=float)
    Jk_b = np.full_like(J, np.nan, dtype=float)
    JL_f = np.full_like(J, np.nan, dtype=float)
    JL_b = np.full_like(J, np.nan, dtype=float)
    diff_k = (J[1:, :] - J[:-1, :]) / grid.dk
    mk = active[:-1, :] & active[1:, :] & np.isfinite(J[:-1, :]) & np.isfinite(J[1:, :])
    Jk_f[:-1, :][mk] = diff_k[mk]
    Jk_b[1:, :][mk] = diff_k[mk]
    diff_L = (J[:, 1:] - J[:, :-1]) / grid.dL
    mL = active[:, :-1] & active[:, 1:] & np.isfinite(J[:, :-1]) & np.isfinite(J[:, 1:])
    JL_f[:, :-1][mL] = diff_L[mL]
    JL_b[:, 1:][mL] = diff_L[mL]
    return Jk_f, Jk_b, JL_f, JL_b


def _build_active_index(active: np.ndarray, grid: _Grid) -> _ActiveIndex:
    idx_full = np.flatnonzero(active.ravel(order="C")).astype(np.int64)
    inv_full = -np.ones(grid.Nx * grid.Ny, dtype=np.int64)
    inv_full[idx_full] = np.arange(idx_full.size, dtype=np.int64)
    return _ActiveIndex(idx_full=idx_full, inv_full=inv_full, n_active=int(idx_full.size))


def _embed_active_to_full(x_act: np.ndarray, act: _ActiveIndex, grid: _Grid, anchor: float = np.nan) -> np.ndarray:
    full = np.full(grid.Nx * grid.Ny, anchor, dtype=float)
    full[act.idx_full] = np.asarray(x_act, dtype=float).reshape(-1)
    return full.reshape(grid.shape, order="C")


def _restrict_full_to_active(x_full: np.ndarray, act: _ActiveIndex) -> np.ndarray:
    return np.asarray(x_full, dtype=float).ravel(order="C")[act.idx_full]


def _build_masked_system(
    grid: _Grid,
    active: np.ndarray,
    node_eval,
    eps_drift: float,
    check_inward: bool = True,
) -> Tuple[sp.csr_matrix, np.ndarray, _ActiveIndex]:
    act = _build_active_index(active, grid)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    f = np.zeros(act.n_active, dtype=float)
    for node in _iter_nodes_where(active):
        i, j = node
        p = int(act.inv_full[_flatten(node, grid)])
        nd = node_eval(node)
        kdot = float(nd.kdot)
        Ldot = float(nd.Ldot)
        flow = float(nd.flow)
        if check_inward and not _inward_one_cell_node(active, node, kdot, Ldot, eps_drift):
            raise RuntimeError(f"Active policy violates inwardness at node={node}.")
        diag = 0.0
        if kdot > eps_drift:
            q = int(act.inv_full[_flatten((i + 1, j), grid)]) if i + 1 < grid.Nx else -1
            if q >= 0:
                rate = kdot / grid.dk
                rows.extend([p, p])
                cols.extend([p, q])
                data.extend([-rate, rate])
                diag -= rate
        elif kdot < -eps_drift:
            q = int(act.inv_full[_flatten((i - 1, j), grid)]) if i - 1 >= 0 else -1
            if q >= 0:
                rate = -kdot / grid.dk
                rows.extend([p, p])
                cols.extend([p, q])
                data.extend([-rate, rate])
                diag -= rate
        if Ldot > eps_drift:
            q = int(act.inv_full[_flatten((i, j + 1), grid)]) if j + 1 < grid.Ny else -1
            if q >= 0:
                rate = Ldot / grid.dL
                rows.extend([p, p])
                cols.extend([p, q])
                data.extend([-rate, rate])
                diag -= rate
        elif Ldot < -eps_drift:
            q = int(act.inv_full[_flatten((i, j - 1), grid)]) if j - 1 >= 0 else -1
            if q >= 0:
                rate = -Ldot / grid.dL
                rows.extend([p, p])
                cols.extend([p, q])
                data.extend([-rate, rate])
                diag -= rate
        if abs(diag) < _EPS:
            rows.append(p)
            cols.append(p)
            data.append(0.0)
        f[p] = flow
    A = sp.coo_matrix((data, (rows, cols)), shape=(act.n_active, act.n_active), dtype=float).tocsr()
    return A, f, act


def _solve_hjb_on_active(A: sp.csr_matrix, f: np.ndarray, rho: float, lam: float = 0.0, J_couple: Optional[np.ndarray] = None) -> np.ndarray:
    n = A.shape[0]
    rhs = np.asarray(f, dtype=float).copy()
    if J_couple is not None:
        rhs += lam * np.asarray(J_couple, dtype=float)
    lhs = (rho + lam) * sp.eye(n, format="csr") - A
    return np.asarray(spla.spsolve(lhs.tocsc(), rhs), dtype=float)


def _project_policy_state_dependent(policy: Dict[str, np.ndarray], grid: _Grid, cal: Calibration, s: int, mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {key: np.asarray(value, dtype=float).copy() for key, value in policy.items()}
    out["tau"][mask] = np.clip(out["tau"][mask], 0.0, cal.tau_max)
    K = np.asarray(grid.k)[:, None]
    LL = np.asarray(grid.L)[None, :]
    w_vec = np.array([_production_block(cal, s, float(k))[1] for k in grid.k], dtype=float)[:, None]
    H_lo = np.maximum(0.0, -LL)
    H_hi = K
    out["H"][mask] = np.minimum(np.maximum(out["H"], H_lo), H_hi)[mask]
    T_lo = np.maximum(cal.transfer_floor, -w_vec + cal.eps_c)
    out["T"][mask] = np.minimum(np.maximum(out["T"], T_lo), cal.transfer_cap)[mask]
    return out


def _blend_and_project(old: Dict[str, np.ndarray], target: Dict[str, np.ndarray], eta: float, grid: _Grid, cal: Calibration, s: int, mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for key in ("tau", "H", "T"):
        out[key] = np.where(mask, (1.0 - eta) * old[key] + eta * target[key], old[key])
    return _project_policy_state_dependent(out, grid, cal, s, mask)


def _policy_supnorm(new: Dict[str, np.ndarray], old: Dict[str, np.ndarray], mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(max(np.nanmax(np.abs(new["tau"][mask] - old["tau"][mask])), np.nanmax(np.abs(new["H"][mask] - old["H"][mask])), np.nanmax(np.abs(new["T"][mask] - old["T"][mask]))))


def _policy_drift_arrays(grid: _Grid, cal: Calibration, s: int, policy: Dict[str, np.ndarray], omega: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kd = np.zeros(grid.shape, dtype=float)
    Ld = np.zeros(grid.shape, dtype=float)
    for node in _iter_nodes_where(mask):
        i, j = node
        _, kd[node], Ld[node] = _node_flow_and_drift(cal, s, float(grid.k[i]), float(grid.L[j]), float(policy["tau"][node]), float(policy["H"][node]), float(policy["T"][node]), float(omega[node]))
    return kd, Ld


def _analytical_transfer_update(Jk_f: np.ndarray, Jk_b: np.ndarray, JL_f: np.ndarray, JL_b: np.ndarray, node: Tuple[int, int], grid: _Grid, cal: Calibration, s: int, tau: float, H: float, omega: np.ndarray) -> float:
    i, j = node
    k = float(grid.k[i])
    L = float(grid.L[j])
    _, w, _, _ = _production_block(cal, s, k)
    _, kdot0, Ldot0 = _node_flow_and_drift(cal, s, k, L, tau, H, 0.0, float(omega[node]), require_feasible=False)
    if not np.isfinite(kdot0) or not np.isfinite(Ldot0):
        return max(cal.transfer_floor, -w + cal.eps_c)
    Jk = 0.0
    JL = 0.0
    if kdot0 > cal.eps_drift and np.isfinite(Jk_f[node]):
        Jk = float(Jk_f[node])
    elif kdot0 < -cal.eps_drift and np.isfinite(Jk_b[node]):
        Jk = float(Jk_b[node])
    if Ldot0 > cal.eps_drift and np.isfinite(JL_f[node]):
        JL = float(JL_f[node])
    elif Ldot0 < -cal.eps_drift and np.isfinite(JL_b[node]):
        JL = float(JL_b[node])
    diff = Jk - JL
    if diff <= _EPS:
        return float(cal.transfer_cap)
    if np.isclose(cal.gamma, 1.0):
        c_w_star = cal.chi / diff
    else:
        c_w_star = (diff / cal.chi) ** (-1.0 / cal.gamma)
    return float(np.clip(c_w_star - w, max(cal.transfer_floor, -w + cal.eps_c), cal.transfer_cap))


def _policy_improvement_gatekeep(grid: _Grid, cal: Calibration, s: int, J: np.ndarray, omega: np.ndarray, active: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    Jk_f, Jk_b, JL_f, JL_b = _masked_upwind_derivatives(J, active, grid)
    target = _empty_policy_like(grid.shape)
    target_mask = np.zeros(grid.shape, dtype=bool)
    tau_grid = np.linspace(0.0, cal.tau_max, cal.tau_points)
    for node in _iter_nodes_where(active):
        i, j = node
        k = float(grid.k[i])
        L = float(grid.L[j])
        _, w, _, _ = _production_block(cal, s, k)
        H_lo = max(0.0, -L)
        H_vals = [float(x) for x in np.linspace(H_lo, k, cal.H_points)]
        best_u: Optional[Tuple[float, float, float]] = None
        best_H = _NEG_INF
        for tau in tau_grid:
            for H in H_vals:
                T_vals = [
                    max(cal.transfer_floor, -w + cal.eps_c),
                    _analytical_transfer_update(Jk_f, Jk_b, JL_f, JL_b, node, grid, cal, s, float(tau), float(H), omega),
                    cal.transfer_cap,
                ]
                for T in T_vals:
                    omg = float(omega[node])
                    if not _static_feasible(cal, s, k, L, float(tau), float(H), float(T), omg):
                        continue
                    flow, kd, Ld = _node_flow_and_drift(cal, s, k, L, float(tau), float(H), float(T), omg)
                    if not _inward_one_cell_node(active, node, kd, Ld, cal.eps_drift):
                        continue
                    Jk = 0.0
                    JL = 0.0
                    if kd > cal.eps_drift:
                        Jk = float(Jk_f[node]) if np.isfinite(Jk_f[node]) else np.nan
                    elif kd < -cal.eps_drift:
                        Jk = float(Jk_b[node]) if np.isfinite(Jk_b[node]) else np.nan
                    if Ld > cal.eps_drift:
                        JL = float(JL_f[node]) if np.isfinite(JL_f[node]) else np.nan
                    elif Ld < -cal.eps_drift:
                        JL = float(JL_b[node]) if np.isfinite(JL_b[node]) else np.nan
                    if (abs(kd) > cal.eps_drift and not np.isfinite(Jk)) or (abs(Ld) > cal.eps_drift and not np.isfinite(JL)):
                        continue
                    H_val = float(flow + kd * Jk + Ld * JL)
                    if H_val > best_H:
                        best_H = H_val
                        best_u = (float(tau), float(H), float(T))
        if best_u is not None:
            target_mask[node] = True
            target["tau"][node], target["H"][node], target["T"][node] = best_u
    return _mask_policy(target, target_mask), target_mask


def _improve_with_prune_closure(grid: _Grid, cal: Calibration, s: int, J: np.ndarray, omega: np.ndarray, active: np.ndarray, max_passes: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    work = np.asarray(active, dtype=bool).copy()
    target = _empty_policy_like(grid.shape)
    if not np.any(work):
        raise RuntimeError("Empty mask entering prune closure.")
    for _ in range(max_passes):
        cand, cand_mask = _policy_improvement_gatekeep(grid, cal, s, J, omega, work)
        new_work = work & cand_mask
        target = cand
        if np.array_equal(new_work, work):
            return _mask_policy(target, work), work
        if not np.any(new_work):
            raise RuntimeError(f"Mask collapsed during prune closure in regime {s}.")
        work = new_work
    return _mask_policy(target, work), work


def _update_private_omega(grid: _Grid, cal: Calibration, s: int, omega_old: np.ndarray, policy: Dict[str, np.ndarray], mask: np.ndarray, omega1_new: Optional[np.ndarray] = None) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return np.asarray(omega_old, dtype=float).copy()

    def node_eval(node: Tuple[int, int]) -> _NodeDriftFlow:
        i, j = node
        _, kd, Ld = _node_flow_and_drift(
            cal,
            s,
            float(grid.k[i]),
            float(grid.L[j]),
            float(policy["tau"][node]),
            float(policy["H"][node]),
            float(policy["T"][node]),
            float(omega_old[node]),
            require_feasible=False,
        )
        flow = float(max(omega_old[node], _EPS) ** (1.0 - cal.gamma))
        if s == 0 and omega1_new is not None:
            flow += cal.lam * float(max(omega1_new[node], _EPS) ** (-cal.gamma))
        return _NodeDriftFlow(kdot=float(kd), Ldot=float(Ld), flow=float(flow))

    A, f, act = _build_masked_system(grid, active, node_eval, cal.eps_drift, check_inward=False)
    D_shift = np.zeros(act.n_active, dtype=float)
    for p_act in range(act.n_active):
        node = _unflatten(int(act.idx_full[p_act]), grid)
        i, j = node
        tau = float(policy["tau"][node])
        H = float(policy["H"][node])
        k = float(grid.k[i])
        L = float(grid.L[j])
        _, _, r_k, sigma_K = _production_block(cal, s, k)
        r_f = _safe_rate(cal, s, k, L, H, tau)
        pi_mc = _private_risky_share(k, L, H)
        R_ce = (1.0 - tau) * r_f + 0.5 * cal.gamma * ((1.0 - tau) ** 2) * (sigma_K ** 2) * (pi_mc ** 2)
        D_shift[p_act] = (1.0 - cal.gamma) * (R_ce - float(omega_old[node]))
    lhs = (cal.rho + (cal.lam if s == 0 else 0.0)) * sp.eye(act.n_active, format="csr") - sp.diags(D_shift, format="csr") - A
    psi_act = np.asarray(spla.spsolve(lhs.tocsc(), f), dtype=float)
    psi_act = np.maximum(psi_act, _EPS)
    psi_full = _embed_active_to_full(psi_act, act, grid, anchor=np.nan)
    omega_new = np.asarray(omega_old, dtype=float).copy()
    valid = active & np.isfinite(psi_full)
    omega_new[valid] = np.maximum(psi_full[valid], _EPS) ** (-1.0 / cal.gamma)
    return omega_new


def _quarantine_fill_nearest(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    out = arr.copy()
    filled = valid_mask.copy()
    if not np.any(filled):
        raise ValueError("Cannot fill from an empty valid mask.")
    for _ in range(arr.shape[0] + arr.shape[1] + 5):
        if filled.all():
            return out
        up = np.zeros_like(filled)
        down = np.zeros_like(filled)
        left = np.zeros_like(filled)
        right = np.zeros_like(filled)
        up[:-1, :] = filled[1:, :]
        down[1:, :] = filled[:-1, :]
        left[:, 1:] = filled[:, :-1]
        right[:, :-1] = filled[:, 1:]
        can_fill = (~filled) & (up | down | left | right)
        if not np.any(can_fill):
            break
        take_up = can_fill & up
        out[take_up] = np.roll(out, -1, axis=0)[take_up]
        filled[take_up] = True
        take_down = can_fill & (~take_up) & down
        out[take_down] = np.roll(out, 1, axis=0)[take_down]
        filled[take_down] = True
        take_left = can_fill & (~take_up) & (~take_down) & left
        out[take_left] = np.roll(out, 1, axis=1)[take_left]
        filled[take_left] = True
        take_right = can_fill & (~take_up) & (~take_down) & (~take_left) & right
        out[take_right] = np.roll(out, -1, axis=1)[take_right]
        filled[take_right] = True
    if not filled.all():
        raise RuntimeError("Disconnected mask prevented omega fill.")
    return out


def _initialize_policy_safe(grid: _Grid, cal: Calibration, s: int, omega: np.ndarray, mask: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    policy = _empty_policy_like(grid.shape)
    stable = np.zeros(grid.shape, dtype=bool)
    work = np.asarray(mask, dtype=bool).copy()
    for node in _iter_nodes_where(work):
        i, j = node
        k = float(grid.k[i])
        L = float(grid.L[j])
        _, w, _, _ = _production_block(cal, s, k)
        best: Optional[Tuple[float, float, float, float]] = None
        for tau, H, T in _coarse_candidates(cal, _build_prim(cal), k, L, w):
            omg = float(omega[node])
            if not _static_feasible(cal, s, k, L, tau, H, T, omg):
                continue
            flow, kd, Ld = _node_flow_and_drift(cal, s, k, L, tau, H, T, omg)
            if not _inward_one_cell_node(work, node, kd, Ld, cal.eps_drift):
                continue
            if best is None or flow > best[0]:
                best = (flow, tau, H, T)
        if best is not None:
            stable[node] = True
            policy["tau"][node] = best[1]
            policy["H"][node] = best[2]
            policy["T"][node] = best[3]
    if not np.any(stable):
        raise RuntimeError(f"Initialization failed: regime {s} mask collapsed.")
    return _mask_policy(policy, stable), stable


def _howard_inner_loop(
    grid: _Grid,
    cal: Calibration,
    omega1: np.ndarray,
    omega0: np.ndarray,
    J1_init: np.ndarray,
    J0_init: np.ndarray,
    u1_init: Dict[str, np.ndarray],
    u0_init: Dict[str, np.ndarray],
    M1_init: np.ndarray,
    M0_init: np.ndarray,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    J1 = np.asarray(J1_init, dtype=float).copy()
    J0 = np.asarray(J0_init, dtype=float).copy()
    u1 = {key: value.copy() for key, value in u1_init.items()}
    u0 = {key: value.copy() for key, value in u0_init.items()}
    M1 = np.asarray(M1_init, dtype=bool).copy()
    M0 = np.asarray(M0_init, dtype=bool).copy()
    M0 &= M1
    for m in range(cal.max_inner):
        def ne1(node: Tuple[int, int]) -> _NodeDriftFlow:
            i, j = node
            flow, kd, Ld = _node_flow_and_drift(cal, 1, float(grid.k[i]), float(grid.L[j]), float(u1["tau"][node]), float(u1["H"][node]), float(u1["T"][node]), float(omega1[node]))
            return _NodeDriftFlow(kdot=kd, Ldot=Ld, flow=flow)

        A1, f1, act1 = _build_masked_system(grid, M1, ne1, cal.eps_drift, check_inward=False)
        J1_new = _embed_active_to_full(_solve_hjb_on_active(A1, f1, cal.rho), act1, grid)

        def ne0(node: Tuple[int, int]) -> _NodeDriftFlow:
            i, j = node
            flow, kd, Ld = _node_flow_and_drift(cal, 0, float(grid.k[i]), float(grid.L[j]), float(u0["tau"][node]), float(u0["H"][node]), float(u0["T"][node]), float(omega0[node]))
            return _NodeDriftFlow(kdot=kd, Ldot=Ld, flow=flow)

        A0, f0, act0 = _build_masked_system(grid, M0, ne0, cal.eps_drift, check_inward=False)
        J0_new = _embed_active_to_full(_solve_hjb_on_active(A0, f0, cal.rho, cal.lam, _restrict_full_to_active(J1_new, act0)), act0, grid)
        try:
            u1_targ, M1_stable = _improve_with_prune_closure(grid, cal, 1, J1_new, omega1, M1)
        except RuntimeError:
            u1_targ, M1_stable = _mask_policy(u1, M1), M1.copy()
        try:
            u0_targ, M0_stable = _improve_with_prune_closure(grid, cal, 0, J0_new, omega0, M0)
        except RuntimeError:
            u0_targ, M0_stable = _mask_policy(u0, M0), M0.copy()
        M0_stable &= M1_stable
        u1_blend = _blend_and_project(u1, _mask_policy(u1_targ, M1_stable), cal.eta_policy, grid, cal, 1, M1_stable)
        u0_blend = _blend_and_project(u0, _mask_policy(u0_targ, M0_stable), cal.eta_policy, grid, cal, 0, M0_stable)
        kdot1, Ldot1 = _policy_drift_arrays(grid, cal, 1, u1_blend, omega1, M1_stable)
        kdot0, Ldot0 = _policy_drift_arrays(grid, cal, 0, u0_blend, omega0, M0_stable)
        ok1 = _inward_one_cell(M1_stable, kdot1, Ldot1, cal.eps_drift)
        ok0 = _inward_one_cell(M0_stable, kdot0, Ldot0, cal.eps_drift)
        u1_next = _empty_policy_like(grid.shape)
        u0_next = _empty_policy_like(grid.shape)
        for key in ("tau", "H", "T"):
            u1_next[key] = np.where(M1_stable, np.where(ok1, u1_blend[key], u1_targ[key]), u1[key])
            u0_next[key] = np.where(M0_stable, np.where(ok0, u0_blend[key], u0_targ[key]), u0[key])
        u1_next = _mask_policy(u1_next, M1_stable)
        u0_next = _mask_policy(u0_next, M0_stable)
        pol_diff = _policy_supnorm(u1_next, u1, M1_stable) + _policy_supnorm(u0_next, u0, M0_stable)
        if verbose:
            print(f"  [Howard] iter {m + 1} pol_diff={pol_diff:.3e}")
        if pol_diff < cal.tol_policy and np.array_equal(M1_stable, M1) and np.array_equal(M0_stable, M0):
            return J1_new, J0_new, u1_next, u0_next, M1_stable, M0_stable
        J1, J0 = J1_new, J0_new
        u1, u0 = u1_next, u0_next
        M1, M0 = M1_stable, M0_stable
    return J1, J0, u1, u0, M1, M0


def _solve_model(cal: Calibration, *, active_omega: bool, verbose: bool) -> Dict[str, Any]:
    grid = _build_grid(cal)
    base_active = _primitive_feasible_mask(grid) & grid.interior_mask
    omega1 = np.full(grid.shape, cal.omega_init, dtype=float)
    omega0 = np.full(grid.shape, cal.omega_init, dtype=float)
    u1, M1 = _initialize_policy_safe(grid, cal, 1, omega1, base_active.copy())
    u0, M0 = _initialize_policy_safe(grid, cal, 0, omega0, base_active.copy() & M1)
    M0 &= M1
    J1 = np.zeros(grid.shape, dtype=float)
    J0 = np.zeros(grid.shape, dtype=float)
    history = []
    M1_prev: Optional[np.ndarray] = None
    M0_prev: Optional[np.ndarray] = None
    stable_count = 0
    for outer in range(cal.max_outer):
        Momega1 = base_active & M1
        Momega0 = base_active & M0
        if active_omega:
            omega1_new = _update_private_omega(grid, cal, 1, omega1, u1, Momega1, None)
            omega0_new = _update_private_omega(grid, cal, 0, omega0, u0, Momega0, omega1_new)
        else:
            omega1_new = omega1.copy()
            omega0_new = omega0.copy()
        omega1_half = (1.0 - cal.zeta_omega) * omega1 + cal.zeta_omega * omega1_new
        omega0_half = (1.0 - cal.zeta_omega) * omega0 + cal.zeta_omega * omega0_new
        omega1_ext = _quarantine_fill_nearest(omega1_half, Momega1)
        omega0_ext = _quarantine_fill_nearest(omega0_half, Momega0)
        V1 = _viability_peel_warm(grid, cal, omega1_ext, 1, base_active, cal.peel_steps)
        V0 = _viability_peel_warm(grid, cal, omega0_ext, 0, base_active, cal.peel_steps) & V1
        V1 &= M1
        V0 &= M0 & V1
        J1_new, J0_new, u1_new, u0_new, M1_new, M0_new = _howard_inner_loop(
            grid,
            cal,
            omega1_ext,
            omega0_ext,
            J1,
            J0,
            u1,
            u0,
            V1,
            V0,
            verbose=verbose,
        )
        core1 = V1 & M1_new
        core0 = V0 & M0_new
        d_omega = 0.0
        if np.any(Momega1):
            d_omega += float(np.nanmax(np.abs(omega1_ext[Momega1] - omega1[Momega1])))
        if np.any(Momega0):
            d_omega += float(np.nanmax(np.abs(omega0_ext[Momega0] - omega0[Momega0])))
        d_J = 0.0
        if np.any(core1):
            d_J += float(np.nanmax(np.abs(J1_new[core1] - J1[core1])))
        if np.any(core0):
            d_J += float(np.nanmax(np.abs(J0_new[core0] - J0[core0])))
        d_u = _policy_supnorm(u1_new, u1, core1) + _policy_supnorm(u0_new, u0, core0)
        history.append(
            {
                "outer_iter": outer + 1,
                "d_omega": d_omega,
                "d_J": d_J,
                "d_u": d_u,
                "size_M1": int(M1_new.sum()),
                "size_M0": int(M0_new.sum()),
            }
        )
        if verbose:
            print(f"[Outer {outer + 1}] d_omega={d_omega:.3e} d_J={d_J:.3e} d_u={d_u:.3e} |M1|={int(M1_new.sum())} |M0|={int(M0_new.sum())}")
        if M1_prev is not None and np.array_equal(M1_new, M1_prev) and np.array_equal(M0_new, M0_prev):
            stable_count += 1
        else:
            stable_count = 0
        M1_prev = M1_new.copy()
        M0_prev = M0_new.copy()
        omega1, omega0 = omega1_ext, omega0_ext
        J1, J0 = J1_new, J0_new
        u1, u0 = u1_new, u0_new
        M1, M0 = M1_new, M0_new
        if d_omega < cal.tol_outer and d_J < cal.tol_outer and d_u < cal.tol_outer and stable_count >= 2:
            break
    return {
        "grid": grid,
        "omega1": omega1,
        "omega0": omega0,
        "J1": J1,
        "J0": J0,
        "u1": u1,
        "u0": u0,
        "M1": M1,
        "M0": M0,
        "history": history,
        "calibration": cal,
        "active_omega": active_omega,
    }


def _log_benchmark_report(solution: Dict[str, Any]) -> Dict[str, float]:
    cal: Calibration = solution["calibration"]
    if not np.isclose(cal.gamma, 1.0):
        raise ValueError("Log benchmark requires gamma=1.")
    grid: _Grid = solution["grid"]
    omega1 = solution["omega1"]
    omega0 = solution["omega0"]
    J1 = solution["J1"]
    J0 = solution["J0"]
    u1 = solution["u1"]
    u0 = solution["u0"]
    M1 = solution["M1"]
    M0 = solution["M0"]
    exact1 = _update_private_omega(grid, cal, 1, omega1, u1, M1, None)
    exact0 = _update_private_omega(grid, cal, 0, omega0, u0, M0, exact1)

    def ne1(node: Tuple[int, int]) -> _NodeDriftFlow:
        i, j = node
        flow, kd, Ld = _node_flow_and_drift(cal, 1, float(grid.k[i]), float(grid.L[j]), float(u1["tau"][node]), float(u1["H"][node]), float(u1["T"][node]), float(exact1[node]))
        return _NodeDriftFlow(kdot=kd, Ldot=Ld, flow=flow)

    def ne0(node: Tuple[int, int]) -> _NodeDriftFlow:
        i, j = node
        flow, kd, Ld = _node_flow_and_drift(cal, 0, float(grid.k[i]), float(grid.L[j]), float(u0["tau"][node]), float(u0["H"][node]), float(u0["T"][node]), float(exact0[node]))
        return _NodeDriftFlow(kdot=kd, Ldot=Ld, flow=flow)

    J1_exact = np.full(grid.shape, np.nan, dtype=float)
    J0_exact = np.full(grid.shape, np.nan, dtype=float)
    if np.any(M1):
        A1, f1, act1 = _build_masked_system(grid, M1, ne1, cal.eps_drift, check_inward=False)
        J1_exact = _embed_active_to_full(_solve_hjb_on_active(A1, f1, cal.rho), act1, grid)
    if np.any(M0):
        A0, f0, act0 = _build_masked_system(grid, M0, ne0, cal.eps_drift, check_inward=False)
        J1_on_M0 = _restrict_full_to_active(J1_exact, act0)
        J0_exact = _embed_active_to_full(_solve_hjb_on_active(A0, f0, cal.rho, cal.lam, J1_on_M0), act0, grid)

    diff1 = float(np.nanmax(np.abs(exact1[M1] - omega1[M1]))) if np.any(M1) else 0.0
    diff0 = float(np.nanmax(np.abs(exact0[M0] - omega0[M0]))) if np.any(M0) else 0.0
    J1_diff = float(np.nanmax(np.abs(J1_exact[M1] - J1[M1]))) if np.any(M1) else 0.0
    J0_diff = float(np.nanmax(np.abs(J0_exact[M0] - J0[M0]))) if np.any(M0) else 0.0
    return {
        "omega1_max_abs_diff": diff1,
        "omega0_max_abs_diff": diff0,
        "J1_max_abs_diff": J1_diff,
        "J0_max_abs_diff": J0_diff,
        "omega1_exact": exact1,
        "omega0_exact": exact0,
        "J1_exact": J1_exact,
        "J0_exact": J0_exact,
    }


def _mid_slice_index(grid: _Grid) -> int:
    return int(np.argmin(np.abs(grid.L)))


def _save_heatmap(x: np.ndarray, y: np.ndarray, z: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(x, y, z.T, shading="auto")
    fig.colorbar(mesh, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("L")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_slice(grid: _Grid, y1: np.ndarray, y0: np.ndarray, title: str, ylabel: str, path: Path) -> None:
    j = _mid_slice_index(grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid.k, y1[:, j], label="Regime 1")
    ax.plot(grid.k, y0[:, j], label="Regime 0")
    ax.set_title(f"{title} at L={grid.L[j]:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_plots(tag: str, solution: Dict[str, Any], output_dir: Path, spec: ExperimentSpec) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid: _Grid = solution["grid"]
    out: Dict[str, str] = {}
    if spec.plot_2d_heatmaps:
        for key, arr in (
            ("omega1", solution["omega1"]),
            ("omega0", solution["omega0"]),
            ("J1", solution["J1"]),
            ("J0", solution["J0"]),
            ("tau1", solution["u1"]["tau"]),
            ("tau0", solution["u0"]["tau"]),
            ("H1", solution["u1"]["H"]),
            ("H0", solution["u0"]["H"]),
            ("T1", solution["u1"]["T"]),
            ("T0", solution["u0"]["T"]),
        ):
            path = output_dir / f"{tag}_{key}.png"
            _save_heatmap(grid.k, grid.L, np.asarray(arr, dtype=float), f"{tag} {key}", path)
            out[key] = str(path)
    if spec.plot_slices:
        for key, y1, y0, ylabel in (
            ("omega_slice", solution["omega1"], solution["omega0"], "omega"),
            ("value_slice", solution["J1"], solution["J0"], "J"),
            ("tau_slice", solution["u1"]["tau"], solution["u0"]["tau"], "tau"),
            ("H_slice", solution["u1"]["H"], solution["u0"]["H"], "H"),
            ("T_slice", solution["u1"]["T"], solution["u0"]["T"], "T"),
        ):
            path = output_dir / f"{tag}_{key}.png"
            _save_slice(grid, np.asarray(y1, dtype=float), np.asarray(y0, dtype=float), f"{tag} {key}", ylabel, path)
            out[key] = str(path)
    return out


def run_experiment(calibration: Calibration, experiment: ExperimentSpec) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(experiment.output_dir).resolve() / ".mplconfig"))
    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"baseline": None, "comparative_statics": {}, "log_benchmark": None}
    if experiment.run_baseline:
        baseline = _solve_model(calibration, active_omega=True, verbose=experiment.verbose)
        baseline["plots"] = _write_plots("baseline", baseline, output_dir, experiment)
        results["baseline"] = baseline
    if experiment.run_comparative_statics:
        for eis in experiment.eis_values:
            gamma = 1.0 / float(eis)
            cal_case = calibration.with_gamma(gamma)
            label = f"eis_{eis:.2f}".replace(".", "p")
            sol = _solve_model(cal_case, active_omega=True, verbose=experiment.verbose)
            sol["plots"] = _write_plots(label, sol, output_dir, experiment)
            results["comparative_statics"][label] = sol
    if experiment.run_log_benchmark:
        log_cal = calibration.with_gamma(1.0)
        log_sol = _solve_model(log_cal, active_omega=True, verbose=experiment.verbose)
        log_sol["plots"] = _write_plots("log_case", log_sol, output_dir, experiment)
        log_sol["benchmark"] = _log_benchmark_report(log_sol)
        results["log_benchmark"] = log_sol
    return results


__all__ = ["Calibration", "ExperimentSpec", "run_experiment"]
