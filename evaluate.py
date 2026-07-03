"""
Local evaluation harness for CS771 Assignment 2, mirroring the official judge:

  my_latent():          returned (w, b, {z_i}) are used to predict TRAIN responses.
  my_latent_updated():  returned (u, a) predict the middle bit of each TEST
                        challenge, then (w, b) predicts the TEST response.

Also computes the Part-5 quantities:
  (a) w vs w~   (b) b vs b~   (c) fraction with 2 z_i - 1 == sign(u~.phi(c_i) + a~)
and, since we generated the data ourselves, the recovery rate of the true
latent middle bits (unavailable on the real dataset, useful as a sanity check).
"""

import time
import numpy as np
from submit import my_latent, my_latent_updated, _phi, _both_embeddings


def main():
    trn = np.loadtxt("train.dat").astype(np.int64)
    tst = np.loadtxt("test.dat").astype(np.int64)

    # ---------------- my_latent ----------------
    tic = time.perf_counter()
    w, b, z = my_latent(trn)
    t1 = time.perf_counter() - tic

    C, r = trn[:, :16], trn[:, 16]
    X0, X1 = _both_embeddings(C)
    Xz = np.where(z[:, None], X1, X0)
    trn_acc = np.mean((Xz @ w + b > 0).astype(int) == r)
    print(f"[my_latent]          time = {t1:7.2f}s   TRAIN acc = {trn_acc:.4f}")

    # ---------------- my_latent_updated ----------------
    tic = time.perf_counter()
    w2, b2, u2, a2 = my_latent_updated(trn)
    t2 = time.perf_counter() - tic

    Ct, rt = tst[:, :16], tst[:, 16]
    z_hat_t = (_phi(Ct) @ u2 + a2 > 0).astype(np.int64)
    X0t, X1t = _both_embeddings(Ct)
    Xht = np.where(z_hat_t[:, None] == 1, X1t, X0t)
    tst_acc = np.mean((Xht @ w2 + b2 > 0).astype(int) == rt)

    z_hat_trn = (_phi(C) @ u2 + a2 > 0).astype(np.int64)
    Xhtr = np.where(z_hat_trn[:, None] == 1, X1, X0)
    trn_acc2 = np.mean((Xhtr @ w2 + b2 > 0).astype(int) == r)
    print(f"[my_latent_updated]  time = {t2:7.2f}s   TRAIN acc = {trn_acc2:.4f}"
          f"   TEST acc = {tst_acc:.4f}")

    # ---------------- Part 5 quantities ----------------
    nw, nw2 = w / np.linalg.norm(w), w2 / np.linalg.norm(w2)
    cos_full = float(nw @ nw2)
    # account for the flip symmetry: negating first 9 coords of w <-> z -> 1-z
    w_flip = nw.copy(); w_flip[:9] *= -1
    cos_flip = float(w_flip @ nw2)
    frac = np.mean((2 * z.astype(int) - 1) == np.sign(_phi(C) @ u2 + a2))
    print(f"[part 5] cos(w, w~) = {cos_full:+.4f}   cos(flip(w), w~) = {cos_flip:+.4f}")
    print(f"[part 5] b = {b:+.4f}   b~ = {b2:+.4f}")
    print(f"[part 5] fraction 2z-1 == sign(u~.phi + a~): {frac:.4f}")

    # ---------------- ground-truth sanity checks (synthetic data only) -------
    try:
        gt = np.load("ground_truth.npz")
        z_true = gt["z_train"]
        m1 = max(np.mean(z == z_true), np.mean(z != z_true))
        m2 = max(np.mean(z_hat_trn == z_true), np.mean(z_hat_trn != z_true))
        print(f"[sanity] my_latent z matches true z (up to global flip): {m1:.4f}")
        print(f"[sanity] updated model's z-hat matches true z (up to flip): {m2:.4f}")
    except FileNotFoundError:
        pass

    np.savez("results.npz", w=w, b=b, z=z, w2=w2, b2=b2, u2=u2, a2=a2,
             t1=t1, t2=t2, trn_acc=trn_acc, trn_acc2=trn_acc2, tst_acc=tst_acc,
             cos_full=cos_full, cos_flip=cos_flip, frac=frac)


if __name__ == "__main__":
    main()
