"""Generates the convergence plot used in the report (electronic, with axis
titles and a legend, as the assignment requires)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from submit import _phi, _both_embeddings, _log_sigmoid, _fix_degenerate, _C_REG, _TOL


def run_alg1_trace(trn, seed=0, iters=15):
    C, r = trn[:, :16], trn[:, 16]
    y = 2 * r - 1
    X0, X1 = _both_embeddings(C)
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=C.shape[0])
    accs = []
    for _ in range(iters):
        Xz = np.where(z[:, None] == 1, X1, X0)
        clf = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500).fit(Xz, r)
        w, b = clf.coef_.ravel(), clf.intercept_[0]
        s0, s1 = X0 @ w + b, X1 @ w + b
        z = (y * s1 > y * s0).astype(np.int64)
        Xz = np.where(z[:, None] == 1, X1, X0)
        accs.append(np.mean((Xz @ w + b > 0).astype(int) == r))
    return accs


def run_alg2_trace(trn, tst, seed=0, iters=15):
    C, r = trn[:, :16], trn[:, 16]
    Ct, rt = tst[:, :16], tst[:, 16]
    y = 2 * r - 1
    X0, X1 = _both_embeddings(C)
    X0t, X1t = _both_embeddings(Ct)
    P, Pt = _phi(C), _phi(Ct)
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=C.shape[0])
    trn_accs, tst_accs = [], []
    for _ in range(iters):
        z = _fix_degenerate(z, rng)
        clf_r = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500).fit(
            np.where(z[:, None] == 1, X1, X0), r)
        w, b = clf_r.coef_.ravel(), clf_r.intercept_[0]
        clf_z = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500).fit(P, z)
        u, a = clf_z.coef_.ravel(), clf_z.intercept_[0]
        t = P @ u + a
        s0, s1 = X0 @ w + b, X1 @ w + b
        ll0 = _log_sigmoid(y * s0) + _log_sigmoid(-t)
        ll1 = _log_sigmoid(y * s1) + _log_sigmoid(+t)
        z = (ll1 > ll0).astype(np.int64)

        zh = (P @ u + a > 0).astype(np.int64)
        Xh = np.where(zh[:, None] == 1, X1, X0)
        trn_accs.append(np.mean((Xh @ w + b > 0).astype(int) == r))
        zht = (Pt @ u + a > 0).astype(np.int64)
        Xht = np.where(zht[:, None] == 1, X1t, X0t)
        tst_accs.append(np.mean((Xht @ w + b > 0).astype(int) == rt))
    return trn_accs, tst_accs


def main():
    trn = np.loadtxt("train.dat").astype(np.int64)
    tst = np.loadtxt("test.dat").astype(np.int64)

    a1 = run_alg1_trace(trn, seed=3)
    a2_trn, a2_tst = run_alg2_trace(trn, tst, seed=5)

    it = np.arange(1, len(a1) + 1)
    plt.figure(figsize=(7, 4.2))
    plt.plot(it, a1, "o-", label="my_latent: train acc (own latents)")
    plt.plot(it, a2_trn, "s-", label="my_latent_updated: train acc (predicted latents)")
    plt.plot(it, a2_tst, "^--", label="my_latent_updated: test acc (predicted latents)")
    plt.xlabel("Alternation number")
    plt.ylabel("Response prediction accuracy")
    plt.title("Convergence of the two alternating optimization algorithms")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=200)
    print("wrote convergence.png")


if __name__ == "__main__":
    main()
