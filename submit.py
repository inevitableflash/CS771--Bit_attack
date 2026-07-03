import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g.
# sklearn.svm etc. You are not allowed to use other libraries such as scipy, keras,
# tensorflow etc.

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_latent AND my_latent_updated BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT

########################################
# Non Editable Region Starting #
########################################
def my_latent( X_trn ):
########################################
#  Non Editable Region Ending  #
########################################

    # X_trn is an (n x 17) matrix: first 16 columns are challenge bits,
    # last column is the response bit.
    C = X_trn[:, :16].astype(np.int64)
    r = X_trn[:, 16].astype(np.int64)
    n = C.shape[0]
    y = 2 * r - 1                                     # {0,1} -> {-1,+1}

    X0, X1 = _both_embeddings(C)                      # phi(I(c,0)), phi(I(c,1))

    rng = np.random.default_rng(0)
    best = (None, None, None, -1.0)

    for restart in range(_N_RESTARTS_1):
        z = rng.integers(0, 2, size=n)                # random latent init
        w, b = None, None

        for it in range(_MAX_ALT_1):
            # ---- (w, b)-step: logistic regression on phi(I(c_i, z_i)) -> r_i
            Xz = np.where(z[:, None] == 1, X1, X0)
            clf = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500)
            clf.fit(Xz, r)
            w, b = clf.coef_.ravel(), clf.intercept_[0]

            # ---- z-step: per-CRP winner-take-all
            s0 = X0 @ w + b
            s1 = X1 @ w + b
            z_new = (y * s1 > y * s0).astype(np.int64)

            if np.array_equal(z_new, z):
                z = z_new
                break
            z = z_new

        # train accuracy of this restart
        Xz = np.where(z[:, None] == 1, X1, X0)
        acc = np.mean((Xz @ w + b > 0).astype(np.int64) == r)
        if acc > best[3]:
            best = (w, b, z, acc)
        if best[3] >= _EARLY_STOP_ACC:
            break

    w, b, z, _ = best
    return w, b, z.astype(bool)


########################################
# Non Editable Region Starting #
########################################
def my_latent_updated( X_trn ):
########################################
#  Non Editable Region Ending  #
########################################

    # X_trn is an (n x 17) matrix: first 16 columns are challenge bits,
    # last column is the response bit.
    C = X_trn[:, :16].astype(np.int64)
    r = X_trn[:, 16].astype(np.int64)
    n = C.shape[0]
    y = 2 * r - 1

    X0, X1 = _both_embeddings(C)                      # 17-bit embeddings
    P = _phi(C)                                       # 16-bit embedding phi(c)

    rng = np.random.default_rng(0)
    best = (None, None, None, None, -1.0)

    for restart in range(_N_RESTARTS_2):
        z = rng.integers(0, 2, size=n)
        w = b = u = a = None

        for it in range(_MAX_ALT_2):
            z = _fix_degenerate(z, rng)

            # ---- (w, b)-step: logistic regression phi(I(c_i, z_i)) -> r_i
            Xz = np.where(z[:, None] == 1, X1, X0)
            clf_r = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500)
            clf_r.fit(Xz, r)
            w, b = clf_r.coef_.ravel(), clf_r.intercept_[0]

            # ---- (u, a)-step: logistic regression phi(c_i) -> z_i
            clf_z = LogisticRegression(C=_C_REG, tol=_TOL, max_iter=500)
            clf_z.fit(P, z)
            u, a = clf_z.coef_.ravel(), clf_z.intercept_[0]

            # ---- z-step: per-CRP winner-take-all on the JOINT likelihood
            s0 = X0 @ w + b
            s1 = X1 @ w + b
            t = P @ u + a
            ll0 = _log_sigmoid(y * s0) + _log_sigmoid(-t)   # z_i = 0
            ll1 = _log_sigmoid(y * s1) + _log_sigmoid(+t)   # z_i = 1
            z_new = (ll1 > ll0).astype(np.int64)

            if np.array_equal(z_new, z):
                z = z_new
                break
            z = z_new

        # Selection criterion mirrors the official evaluation pipeline:
        # predict middle bit with (u, a), then predict response with (w, b)
        z_hat = (P @ u + a > 0).astype(np.int64)
        Xh = np.where(z_hat[:, None] == 1, X1, X0)
        acc = np.mean((Xh @ w + b > 0).astype(np.int64) == r)
        if acc > best[4]:
            best = (w, b, u, a, acc)
        if best[4] >= _EARLY_STOP_ACC:
            break

    w, b, u, a, _ = best
    return w, b, u, a


################################
#  Helper functions and hyperparameters (freely editable region)  #
################################

_C_REG = 20.0          # inverse L2 regularization strength for logistic regression
_TOL = 1e-3            # solver tolerance
_MAX_ALT_1 = 30        # max alternations, my_latent
_MAX_ALT_2 = 40        # max alternations, my_latent_updated
_N_RESTARTS_1 = 6      # random restarts, my_latent
_N_RESTARTS_2 = 8      # random restarts, my_latent_updated
_EARLY_STOP_ACC = 0.999


def _phi(C):
    """Standard arbiter-PUF embedding phi_i(c) = prod_{j >= i} (1 - 2 c_j).

    C is an (n x k) 0/1 matrix; returns an (n x k) +/-1 matrix.
    """
    D = 1 - 2 * C
    return np.cumprod(D[:, ::-1], axis=1)[:, ::-1].astype(np.float64)


def _both_embeddings(C):
    """Return (phi(I(c,0)), phi(I(c,1))) for all rows c of C, exploiting the
    fact that flipping the inserted middle bit (position 8, 0-indexed) only
    negates the first 9 coordinates of the 17-dim embedding."""
    n = C.shape[0]
    C17 = np.concatenate([C[:, :8], np.zeros((n, 1), dtype=np.int64), C[:, 8:]],
                         axis=1)
    X0 = _phi(C17)
    X1 = X0.copy()
    X1[:, :9] *= -1.0
    return X0, X1


def _log_sigmoid(x):
    """Numerically stable log(sigmoid(x)) using only numpy."""
    return -np.logaddexp(0.0, -x)


def _fix_degenerate(z, rng):
    """Logistic regression needs both classes present; if the latent vector
    collapses to a single value, re-seed a small random fraction."""
    if z.min() == z.max():
        idx = rng.choice(z.shape[0], size=max(1, z.shape[0] // 20),
                         replace=False)
        z = z.copy()
        z[idx] = 1 - z[idx]
    return z
