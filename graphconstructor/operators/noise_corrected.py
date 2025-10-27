from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from ..graph import Graph
from .base import GraphOperator


@dataclass(slots=True)
class NoiseCorrected(GraphOperator):
    """
    Noise-Corrected (NC) backbone per Coscia & Neffke (2017), "Network Backboning with Noisy Data".

    Keeps edge (i,j) if L_ij >= δ * sqrt( Var[ L_ij ] ),
    where L is the symmetric lift and Var is obtained via a binomial model
    with a Beta prior calibrated from hypergeometric-inspired moments.

    Parameters
    ----------
    delta : float
        Threshold in standard deviations (typical: 1.28 ~10%, 1.64 ~5%, 2.32 ~1.2%).
    derivative : {"constant","full"}
        Delta-method derivative:
        - "constant": g'(n) = 2*kappa / (kappa*n + 1)^2  (hold margins fixed)
        - "full":     g'(n) = 2*(kappa + n*dκ/dn) / (kappa*n + 1)^2, where
                      dκ/dn = 1/(Ni*Nj) - Ntot*(Ni+Nj)/(Ni*Nj)^2
    copy_meta : bool
        If True, copy metadata DataFrame; otherwise keep reference.
    """
    delta: float = 1.64
    derivative: str = "constant"
    copy_meta: bool = True
    supported_modes = ["similarity"]

    # ---------- Bayesian shrinkage for P_ij ----------
    def _posterior_mean_p(
        self, nij: np.ndarray, Ni: np.ndarray, Nj: np.ndarray, Ntot: float
    ) -> np.ndarray:
        eps = 1e-12
        Ntot_safe = max(Ntot, 1.0)

        # Hypergeometric-inspired prior moments
        mu = (Ni * Nj) / (Ntot_safe * Ntot_safe)
        denom = (Ntot_safe * Ntot_safe * (Ntot_safe - 1.0))
        var = np.zeros_like(mu, dtype=float)
        mask = denom > 0
        if np.any(mask):
            var[mask] = (
                Ni[mask]
                * Nj[mask]
                * (Ntot_safe - Ni[mask])
                * (Ntot_safe - Nj[mask])
            ) / (denom * (Ntot_safe * Ntot_safe))

        # Guardrails
        var = np.clip(var, eps, None)
        mu = np.clip(mu, eps, 1.0 - eps)

        # Convert to Beta(α,β)
        alpha = (mu * mu / var) * (1.0 - mu) - mu
        beta = mu * (((1.0 - mu) * (1.0 - mu)) / var + 1.0) - 1.0
        alpha = np.clip(alpha, eps, None)
        beta = np.clip(beta, eps, None)

        # Posterior mean E[P|data]
        return (nij + alpha) / (Ntot_safe + alpha + beta)

    # ---------- Core mask computation ----------
    def _compute_keep_mask(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        w: np.ndarray,
        row_sum: np.ndarray,
        col_sum: np.ndarray,
        Ntot: float,
    ) -> np.ndarray:
        eps = 1e-18

        # Expected count & kappa
        E = (row_sum[rows] * col_sum[cols]) / max(Ntot, 1.0)
        E = np.clip(E, eps, None)
        kappa = 1.0 / E  # == Ntot / (Ni * Nj)

        # Symmetric lift
        kn = kappa * w
        Ltilde = (kn - 1.0) / (kn + 1.0)

        # Posterior mean and Var[N]
        p_post = self._posterior_mean_p(w, row_sum[rows], col_sum[cols], float(Ntot))
        p_post = np.clip(p_post, 1e-15, 1.0 - 1e-15)
        VarN = float(Ntot) * p_post * (1.0 - p_post)

        # Delta-method derivative
        if self.derivative == "constant":
            gprime = (2.0 * kappa) / ((kappa * w + 1.0) ** 2)
        elif self.derivative == "full":
            Ni = row_sum[rows]
            Nj = col_sum[cols]
            den = np.clip(Ni * Nj, eps, None)
            d_kappa = (1.0 / den) - (float(Ntot) * (Ni + Nj)) / (den * den)
            gprime = (2.0 * (kappa + w * d_kappa)) / ((kappa * w + 1.0) ** 2)
        else:
            raise ValueError("derivative must be 'constant' or 'full'.")

        VarL = (gprime ** 2) * VarN
        thresh = self.delta * np.sqrt(VarL)

        return Ltilde >= thresh

    # ---------- Directed / Undirected pathways ----------
    def _apply_directed(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("NoiseCorrected requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(
                A.copy(),
                directed=True,
                weighted=G.weighted,
                mode=G.mode,
                meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            )

        row_sum = np.asarray(A.sum(axis=1)).ravel()
        col_sum = np.asarray(A.sum(axis=0)).ravel()
        Ntot = float(row_sum.sum())

        coo = A.tocoo()
        rows, cols, w = coo.row, coo.col, coo.data

        keep = self._compute_keep_mask(rows, cols, w, row_sum, col_sum, Ntot)
        A_f = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=A.shape)

        return Graph.from_csr(
            A_f,
            directed=True,
            weighted=G.weighted,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
        )

    def _apply_undirected(self, G: Graph) -> Graph:
        A = G.adj.tocsr(copy=False)
        if (A.data < 0).any():
            raise ValueError("NoiseCorrected requires nonnegative weights.")
        if A.nnz == 0:
            return Graph.from_csr(
                A.copy(),
                directed=False,
                weighted=G.weighted,
                mode=G.mode,
                meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
                sym_op="max",
            )

        Au = sp.triu(A, k=1).tocoo()
        rows, cols, w = Au.row, Au.col, Au.data

        row_sum = np.asarray(A.sum(axis=1)).ravel()
        # symmetric case: col_sum == row_sum
        Ntot = float(row_sum.sum())

        keep = self._compute_keep_mask(rows, cols, w, row_sum, row_sum, Ntot)

        r = rows[keep]
        c = cols[keep]
        data = w[keep]
        rows_full = np.concatenate([r, c])
        cols_full = np.concatenate([c, r])
        data_full = np.concatenate([data, data])
        A_f = sp.csr_matrix((data_full, (rows_full, cols_full)), shape=A.shape)

        return Graph.from_csr(
            A_f,
            directed=False,
            weighted=G.weighted,
            mode=G.mode,
            meta=(G.meta.copy() if (self.copy_meta and G.meta is not None) else G.meta),
            sym_op="max",
        )

    def apply(self, G: Graph) -> Graph:
        self._check_mode_supported(G)
        return self._apply_directed(G) if G.directed else self._apply_undirected(G)
